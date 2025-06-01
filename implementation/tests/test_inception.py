import pytest
import torch

from models.inception.inception_parts import ConvolutionBlock, InceptionBlockNaive, InceptionBlockNaivePimped, InceptionBlockV1, InceptionBlockV3_F5, InceptionBlockV3_F7

@pytest.fixture
def image() -> torch.Tensor:
    return torch.randint(0, 255, size=(5, 192, 28, 28)).float()


def test_convblock(image):
    B, C, H, W = image.size()
    OUT_CHANNELS = 6
    block = ConvolutionBlock(C, OUT_CHANNELS, kernel_size=1)
    output = block(image)
    assert output.size() == (B, OUT_CHANNELS, H, W)

def test_inception_block_naive(image):
    B, C, H, W = image.size()
    RED_1x1, RED_3x3, RED_5x5 = 64, 128, 32
    block = InceptionBlockNaive(in_channels=C, red_1x1=RED_1x1, red_3x3=RED_3x3, red_5x5=RED_5x5)
    output = block(image)
    assert output.size() == (B, RED_1x1 + RED_3x3 + RED_5x5 + C, H, W)


def test_inception_block_naive_pimped(image):
    B, C, H, W = image.size()
    RED_1x1, RED_3x3, RED_5x5 = 64, 128, 32
    block = InceptionBlockNaivePimped(in_channels=C, red_1x1=RED_1x1, red_3x3=RED_3x3, red_5x5=RED_5x5)
    output = block(image)
    assert output.size() == (B, RED_1x1 + RED_3x3 + RED_5x5 + C, H, W)


def test_inception_blockv1(image):
    B, C, H, W = image.size()
    RED_1x1, RED_3x3, OUT_3x3, RED_5x5, OUT_5x5, OUT_POOL = 64, 96, 128, 16, 32, 32
    block = InceptionBlockV1(in_channels=C, red_1x1=RED_1x1, 
                             red_3x3=RED_3x3, out3x3=OUT_3x3,
                             red_5x5=RED_5x5, out_5x5=OUT_5x5, 
                             out_pool=OUT_POOL)
    output = block(image)
    assert output.size() == (B, RED_1x1 + OUT_3x3 + OUT_5x5 + OUT_POOL, H, W)


def test_inception_block_f5(image):
    B, C, H, W = image.size()
    RED_DOUBLE_3x3, MID_DOUBLE_3x3, OUT_DOUBLE_3x3, RED_3x3, OUT_3x3, RED_POOL, RED_1x1 = 16, 24, 32, 96, 128, 32, 64
    block = InceptionBlockV3_F5(in_channels=C, 
                                red_double_3x3=RED_DOUBLE_3x3, 
                                mid_double_3x3=MID_DOUBLE_3x3, 
                                out_double_3x3=OUT_DOUBLE_3x3, 
                                red_3x3=RED_3x3, out_3x3=OUT_3x3, 
                                red_pool=RED_POOL, red_1x1=RED_1x1
                                )
    output = block(image)
    assert output.size() == (B, OUT_DOUBLE_3x3 + OUT_3x3 + RED_POOL + RED_1x1, H, W)


def test_inception_block_f7(image):
    B, C, H, W = image.size()
    RED_SPLIT_3x3, CONV_SPLIT_3x3, OUT_SPLIT_3x3_1x3, OUT_SPLIT_3x3_3x1 = 16, 32, 16, 16
    RED_SPLIT_1x1, OUT_SPLIT_1x1_1x3, OUT_SPLIT_1x1_3x1 = 96, 64, 64
    RED_POOL, RED_1x1 = 32, 32

    block = InceptionBlockV3_F7(in_channels=C, 
                                red_split_3x3=RED_SPLIT_3x3, conv_split_3x3=CONV_SPLIT_3x3, 
                                out_split_3x3_1x3=OUT_SPLIT_3x3_1x3, out_split_3x3_3x1=OUT_SPLIT_3x3_3x1, 
                                red_split_1x1=RED_SPLIT_1x1, 
                                out_split_1x1_1x3=OUT_SPLIT_1x1_1x3, out_split_1x1_3x1=OUT_SPLIT_1x1_3x1, 
                                red_pool=RED_POOL, red_1x1=RED_1x1
                                )
    output = block(image)
    assert output.size() == (B, OUT_SPLIT_3x3_1x3 + OUT_SPLIT_3x3_3x1 + 
                             OUT_SPLIT_1x1_1x3 + OUT_SPLIT_1x1_3x1 +
                             RED_POOL + RED_1x1, H, W)



