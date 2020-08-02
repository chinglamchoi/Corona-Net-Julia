using Flux
using Flux: throttle, logitbinarycrossentropy
using Base.Iterators: partition
using CSV
using Images
using Statistics: mean
using BSON
using CuArrays, CUDAnative, CUDAdrv
CuArrays.allowscalar(false)
include("unet_flux2.jl")

img_train_path, img_test_path = "./data/train_imgs/", "./data/test_imgs/"
mask_train_path, mask_test_path = "./data/train_msks/", "./data/test_msks/"

train_size, test_size = 1920, 20

mb_size = 8
mb_idxs = collect(partition(1:train_size, mb_size))

function load_me(img_path, mask_path, idxs)
    y1 = Array{Float32}(undef, 512, 512, 1, length(idxs))
    y2 = Array{Float32}(undef, 512, 512, 3, length(idxs))
    cnt1 = 1
    for i in idxs
        y1[:, :, 1, cnt1] = Float32.(permutedims(channelview(load(img_path*string(cnt1-1)*".jpg")), (2, 1)))
	y2[:, :, :, cnt1] = Float32.(permutedims(channelview(load(mask_path*string(cnt1-1)*".jpg")), (3,2, 1)))
        cnt1 += 1
    end
    return [(y1, y2)]
end

UNet_model = UNet() |> gpu

function accuracy(x,y)
    y_hat = σ.(UNet_model(x))
    return 2 * sum(y_hat .* y) / (sum(y_hat) + sum(y))
end

optimiser = ADAM(0.01, (0.9, 0.999))
best_acc, last_improve, epoch_num, threshold = 0.0, 0, 300, 0.95


function my_custom_train!(ps, data, opt)
    ps = params(ps)
    gs = gradient(ps) do
        global training_loss = mean(logitbinarycrossentropy.(UNet_model(data[1][1]), data[1][2]))
        return training_loss
    end
    Flux.update!(opt, ps, gs)
    return training_loss
end

loss(x,y) = mean(logitbinarycrossentropy.(UNet_model(x) |> gpu, y |> gpu))

last_improve = 0
#BSON.@load "best_model.BSON" UNet_model

for i in 1:epoch_num
    epoch_loss, cnt = 0.0, 0
    if last_improve >= 15
	global optimiser.eta /= 10
	global last_improve = 0
    end
    for o in 1:length(mb_idxs)
        train_batch = gpu.(load_me(img_train_path, mask_train_path, mb_idxs[o]))
        Flux.train!(loss, params(UNet_model), train_batch, optimiser)
	cnt += 1
    end
    epoch_loss /= cnt
    testset = gpu.(load_me(img_test_path, mask_test_path, collect(1:test_size)))
    acc = accuracy((testset...)...)
    if acc > best_acc
        BSON.@save "best_model.BSON" UNet_model
        global best_acc = acc
        println("New best accuracy!")
    else
	global last_improve += 1
    end
    println("Epoch ", i, ": ", acc, "\n")
end
