using Flux

model = Chain(
    # First set of convolutional layers
    Conv((3, 3), 3=>32, relu),   
    MaxPool((2, 2)),

    # Second set of convolutional layers
    Conv((3, 3), 32=>64, relu),  # 32 input channels, 64 filters
    MaxPool((2, 2)),

    # Third set of convolutional layers
    Conv((3, 3), 64=>128, relu), # 64 input channels, 128 filters
    MaxPool((2, 2)),

    # Fourth set of convolutional layers
    Conv((3, 3), 128=>256, relu),# 128 input channels, 256 filters
    MaxPool((2, 2)),

    # Fifth set of convolutional layers
    Conv((3, 3), 256=>512, relu),# 256 input channels, 512 filters
    MaxPool((2, 2)),

    # Flatten the output before feeding into fully connected layers
    Flux.flatten,

    # Fully connected layers (Dense layers)
    Dense(512 * 7 * 7, 1024, relu),   
    Dense(1024, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 10),  
    softmax         
)

println(model)
loss(x, y) = Flux.crossentropy(model(x), y)

# Define an optimizer
opt = Adam()

X_train = rand(Float32, 224, 224, 3, 1000)  
y_train = Flux.onehotbatch(rand(1:10, 1000), 1:10)  

# Training loop
for epoch in 1:10
    for i in 1:size(X_train, 4)
        x = X_train[:, :, :, i]  
        y = y_train[:, i]          
        l = loss(x, y)
        Flux.back!(l)
        Flux.Optimise.update!(opt, model)

        println("Epoch $epoch, Iteration $i, Loss: $l")
    end
end
