require 'nn'

torch.manualSeed(0)

function getModel(opt)
    local model = nn.Sequential()
    model:add(nn.Reshape(28*28))
    model:add(nn.Linear(28*28, 100))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(100, 10))
    model:add(nn.LogSoftMax())
    
    if opt.gpu == 1 then model = model:cuda() end
    return model
end
