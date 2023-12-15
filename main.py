import math
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import tracker, lab, monit, experiment
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.stylegan import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.utils import cycle_dataloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, image_size: int):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()
    discriminator: Discriminator
    generator: Generator
    mapping_network: MappingNetwork
    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam
    gradient_penalty = GradientPenalty()
    gradient_penalty_coefficient: float = 10.
    path_length_penalty: PathLengthPenalty
    loader: Iterator
    batch_size: int = 32
    d_latent: int = 512
    image_size: int = 32
    mapping_network_layers: int = 8
    learning_rate: float = 1e-3
    mapping_network_learning_rate: float = 1e-5
    gradient_accumulate_steps: int = 1
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    style_mixing_prob: float = 0.9
    training_steps: int = 150_000
    n_gen_blocks: int
    lazy_gradient_penalty_interval: int = 4
    lazy_path_penalty_interval: int = 32
    lazy_path_penalty_after: int = 5_000
    log_generated_interval: int = 500
    save_checkpoint_interval: int = 2_000
    mode: ModeState
    log_layer_outputs: bool = False
    dataset_path: str = str(lab.get_data_path() / 'stylegan2')

    def init(self):
        dataset = Dataset(self.dataset_path, self.image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        self.loader = cycle_dataloader(dataloader)
        log_resolution = int(math.log2(self.image_size))
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)
        if self.log_layer_outputs:
            hook_model_outputs(self.mode, self.discriminator, 'discriminator')
            hook_model_outputs(self.mode, self.generator, 'generator')
            hook_model_outputs(self.mode, self.mapping_network, 'mapping_network')
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )
        tracker.set_image("generated", True)

    def get_w(self, batch_size: int):
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_noise(self, batch_size: int):
        noise = []
        resolution = 4
        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
                n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
                noise.append((n1, n2))
                resolution *= 2
        return noise

    def generate_images(self, batch_size: int):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)
        images = self.generator(w, noise)
        return images, w

    def step(self, idx: int):
        with monit.section('Discriminator'):
            self.discriminator_optimizer.zero_grad()
            for i in range(self.gradient_accumulate_steps):
                with self.mode.update(is_log_activations=(idx + 1) % self.log_generated_interval == 0):
                    generated_images, _ = self.generate_images(self.batch_size)
                    fake_output = self.discriminator(generated_images.detach())
                    real_images = next(self.loader).to(self.device)
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()
                    real_output = self.discriminator(real_images)
                    real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                    disc_loss = real_loss + fake_loss
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add('loss.gp', gp)
                        disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval
                    disc_loss.backward()
                    tracker.add('loss.discriminator', disc_loss)
            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('discriminator', self.discriminator)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.discriminator_optimizer.step()
        with monit.section('Generator'):
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()
            for i in range(self.gradient_accumulate_steps):
                generated_images, w = self.generate_images(self.batch_size)
                fake_output = self.discriminator(generated_images)
                gen_loss = self.generator_loss(fake_output)
                if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                    plp = self.path_length_penalty(w, generated_images)
                    if not torch.isnan(plp):
                        tracker.add('loss.plp', plp)
                        gen_loss = gen_loss + plp
                    gen_loss.backward()
                    tracker.add('loss.generator', gen_loss)
                if (idx + 1) % self.log_generated_interval == 0:
                    tracker.add('generator', self.generator)
                    tracker.add('mapping_network', self.mapping_network)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)
                self.generator_optimizer.step()
                self.mapping_network_optimizer.step()
            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('generated', torch.cat([generated_images[:6], real_images[:3]], dim=0))
            if (idx + 1) % self.save_checkpoint_interval == 0:
                experiment.save_checkpoint()
            tracker.save()

    def train(self):
        for i in monit.loop(self.training_steps):
            self.step(i)
            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()


def main():
    experiment.create(name='stylegan2')
    configs = Configs()
    experiment.configs(configs, {
        'device.cuda_device': 0,
        'image_size': 64,
        'log_generated_interval': 200
    })
    configs.init()
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)
    with experiment.start():
        configs.train()


if __name__ == '__main__':
    main()
