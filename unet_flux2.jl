using Flux
using CuArrays, CUDAnative
CuArrays.allowscalar(false)

block1(in_channels, features) = Chain(Conv((3,3), in_channels=>features, pad=1),
    BatchNorm(features, relu), #calls n-1'th dim
    Conv((3,3), features=>features, pad=1),
    BatchNorm(features, relu)) |> gpu

block2(in_channels, features) = Chain(MaxPool((2,2), stride=2),
	Conv((3,3), in_channels=>features, pad=1),
    BatchNorm(features, relu), #calls n-1'th dim
    Conv((3,3), features=>features, pad=1),
    BatchNorm(features, relu)) |> gpu


upconv(in_channels, features) = ConvTranspose((2,2), in_channels=>features, stride=2) |> gpu

conv(in_channels, out_channels) = Conv((1,1), in_channels=>out_channels) |> gpu

struct UNet
    conv_block
    conv_block2
    bottle
    upconv_block
    conv_
end

function UNet()
    conv_block = (block1(3, 32), block2(32, 32*2), block2(32*2, 32*4), block2(32*4, 32*8))
    conv_block2 = (block1(32*16, 32*8), block1(32*8, 32*4), block1(32*4, 32*2), block1(32*2, 32))
    bottle = block2(32*8, 32*16)
    upconv_block = (upconv(32*16, 32*8), upconv(32*8, 32*4), upconv(32*4, 32*2), upconv(32*2, 32))
    conv_ = conv(32, 1)
    UNet(conv_block, conv_block2, bottle, upconv_block, conv_) |> gpu
end


function (u::UNet)(x)
    enc1 = u.conv_block[1](x) |> gpu
    enc2 = u.conv_block[2](enc1) |> gpu
    enc3 = u.conv_block[3](enc2) |> gpu
    enc4 = u.conv_block[4](enc3) |> gpu
    
    bn = u.bottle(enc4) |> gpu
	
    dec4 = u.upconv_block[1](bn) |> gpu
    dec4 = cat(dims=3, dec4, enc4) |> gpu
    dec4 = u.conv_block2[1](dec4) |> gpu
    dec3 = u.upconv_block[2](dec4) |> gpu
    dec3 = cat(dims=3, dec3, enc3) |> gpu
    dec3 = u.conv_block2[2](dec3) |> gpu
    dec2 = u.upconv_block[3](dec3) |> gpu
    dec2 = cat(dims=3, dec2, enc2) |> gpu
    dec2 = u.conv_block2[3](dec2) |> gpu
    dec1 = u.upconv_block[4](dec2) |> gpu
    dec1 = cat(dims=3, dec1, enc1) |> gpu
    dec1 = u.conv_block2[4](dec1) |> gpu
    dec1 = u.conv_(dec1) |> gpu
end
