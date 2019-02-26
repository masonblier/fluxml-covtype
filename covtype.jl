using Flux, Statistics, DelimitedFiles
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

# covtype.jl
println("covtype.jl")

# simple dense network with fluxml to reproduce covtype dataset results
# https://archive.ics.uci.edu/ml/datasets/covertype
# based on mlp.jl and housing.jl examples from fluxml/model-zoo

# ADAM hyperparamters, cranked up for fast/loose learning
adam_η = 0.05
adam_β = (0.8, 0.999)

# dense network with softmax output
m = Chain(
  Dense(54, 14, relu),
  Dense(14, 7),
  softmax)

# loss / accuracy / optimizer fns
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
optimizer = ADAM(adam_η, adam_β)

# load data
println("loading data")

# training and validation counts from original study
training_count = 11340
validation_count = 3780

# training batch size/count
batch_size = 20 # repeated dataset per training
batch_count = 10 # number of trainings

# read csv data
rawdata = readdlm("covtype-min.data", ',')
data = rawdata[1:(training_count + validation_count),:]

# in-line normalize datasets to 0.0-1.0, assume 0-intercept
normalize(x) = x ./ maximum(x)

# split column sets to normalize seperately
elevation_meters = normalize(data[:, 1:1])
aspect_azimuth = normalize(data[:, 2:2])
slope_degrees = normalize(data[:, 3:3])
horizontal_distances = normalize(hcat(data[:, 4:6], data[:, 10:10]))
hillshades = normalize(data[:, 7:9])
binary_columns = normalize(data[:, 11:54])

# input data
x = hcat(elevation_meters,aspect_azimuth,slope_degrees,horizontal_distances,
    hillshades,binary_columns)'
training_set_x = x[:,1:training_count]
test_set_x = x[:,(1 + training_count):(training_count + validation_count)]

println("  training input", size(training_set_x), "; test input", size(test_set_x))

# last column is target Cover_type 1-7
training_set_y = vec(data[1:training_count, 55:55]')
test_set_y = vec(data[(1 + training_count):(training_count + validation_count), 55:55]')
training_onehot_y = onehotbatch(training_set_y, 1:7)
test_onehot_y = onehotbatch(test_set_y, 1:7)


# train
println("training")

# train batch_count times
for i in 1:batch_count
  # entire dataset repeated batch_size times
  dataset = repeated((training_set_x, training_onehot_y), batch_size)
  Flux.train!(loss, params(m), dataset, optimizer)
  println("  ", accuracy(training_set_x, training_onehot_y), " ($i)")
end

# validate
println("validation")
println("  ", accuracy(test_set_x, test_onehot_y))

# sample of actual/expected output
println("covtype", floor.(Int, test_set_y[1:20]))
println("guess  ", onecold(m(test_set_x[:,1:20])))
