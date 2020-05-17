using Flux
using CuArrays
CuArrays.allowscalar(false)

# WHCN
function block(y, in_channels, features)
	chain = Chain(
		Conv((3,3), in_channels=>features, pad=1),
		BatchNorm(features, relu), #calls n-1'th dim
		Conv((3,3), features=>features, pad=1),
		BatchNorm(features, relu)
		) |> gpu
	return chain(y)
end

function encode(y, in_channels, features)
	chain = Chain(
		MaxPool((2,2), stride=2),
		x -> block(x, in_channels, features)
		) |> gpu
	return chain(y)
end


function upconv(y, in_channels, features)
	m = ConvTranspose((2,2), in_channels=>features, stride=2) |> gpu
	return m(y)
end

function UNet(y, in_channels=3, out_channels=1, init_features=32)
	encoder1 = x -> block(x, in_channels, init_features)
	encoder2 = x -> encode(x, init_features, init_features*2)
	encoder3 = x -> encode(x, init_features*2, init_features*4)
	encoder4 = x -> encode(x, init_features*4, init_features*8)

	bottleneck = x -> encode(x, init_features*8, init_features*16)

	upconv4 = x -> upconv(x, init_features*16, init_features*8)
	decoder4 = x -> block(x, init_features*16, init_features*8)
	upconv3 = x -> upconv(x, init_features*8, init_features*4)
	decoder3 = x -> block(x, init_features*8, init_features*4)
	upconv2 = x -> upconv(x, init_features*4, init_features*2)
	decoder2 = x -> block(x, init_features*4, init_features*2)
	upconv1 = x -> upconv(x, init_features*2, init_features)
	decoder1 = x -> block(x, init_features*2, init_features)
	conv = Conv((1,1), init_features=>out_channels) |> gpu

	enc1 = encoder1(y)
	enc2 = encoder2(enc1)
	enc3 = encoder3(enc2)
	enc4 = encoder4(enc3)

	bn = bottleneck(enc4)

	dec4 = upconv4(bn)
	dec4 = cat(dims=3, dec4, enc4) |> gpu
	dec4 = decoder4(dec4)

	dec3 = upconv3(dec4)
	dec3 = cat(dims=3, dec3, enc3) |> gpu
	dec3 = decoder3(dec3)

	dec2 = upconv2(dec3)
	dec2 = cat(dims=3, dec2, enc2) |> gpu
	dec2 = decoder2(dec2)

	dec1 = upconv1(dec2)
	dec1 = cat(dims=3, dec1, enc1) |> gpu
	dec1 = decoder1(dec1)

	dec1 = conv(dec1)

	return dec1
end

UNet = gpu(UNet)
