require 'model'
require 'dataset'
require 'train'
require 'evaluate'

torch.setdefaulttensortype('torch.FloatTensor')

---------------------------------------------------------------------------
-------------------- Parse command-line parameters ------------------------
---------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text("Training model for classifying MNIST")
cmd:text()
cmd:text("Options")
cmd:option("-gpu", 0, "set this flag to 1 if you want to use GPU")
cmd:option("-batchSize", 200, "the batch size for training")
cmd:option("-numEpochs", 100, "the number of epochs to train")

opt = cmd:parse(arg)
if opt.gpu == 1 then
    cunnOk, cunn = pcall(require, "cunn")
    cutorchOk, cutorch = pcall(require, "cutorch")
    
    if not cunnOk or not cutorchOk then
        print ("cunn and/or cutorch are not properly configured.")
        print ("Falling back to CPU mode...")
        opt.gpu = 0
    end
end

---------------------------------------------------------------------------
------------------------ Loading net and dataset --------------------------
---------------------------------------------------------------------------
print ("\n[ Loading network ]")
net = getModel(opt)
params, gradParams = net:getParameters()
print ("\t# params = " .. params:size()[1])

print ("\n[ Loading dataset ]")
dataset, opt.trainingMean = getDataset()
print ("\t#training images = " .. dataset['trainset'].size)
print ("\t#validation images = " .. dataset['validationset'].size)
print ("\t#test images = " .. dataset['testset'].size)
print ("\t dataset mean = " .. opt.trainingMean)

---------------------------------------------------------------------------
------------------------ Train model on dataset ---------------------------
---------------------------------------------------------------------------

print ("\n[ Start training ]")
opt.criterion = nn.ClassNLLCriterion()
opt.sgd_params = {
    learningRate = 1e-2,
    weightDecay = 0.0005,
    learningRateDecay = 1e-4,
    momentum = 0.9
}

trainLoss, trainAcc = evaluate(opt, net, dataset['trainset'])
valLoss, valAcc = evaluate(opt, net, dataset['validationset'])
testLoss, testAcc = evaluate(opt, net, dataset['testset'])
print ("\tepoch 0/" .. opt.numEpochs .. ": trainLoss = " .. trainLoss .. ", trainAcc = " .. trainAcc .. ", valAcc = " .. valAcc .. ", testAcc = " .. testAcc)

opt.itersInEpoch = math.ceil(dataset['trainset'].size / opt.batchSize)
for epochCtr = 1, opt.numEpochs do
    local trainLoss = trainEpoch(opt, net, params, gradParams, dataset)
    local _, trainAcc = evaluate(opt, net, dataset['trainset'])
    local valLoss, valAcc = evaluate(opt, net, dataset['validationset'])
    local _, testAcc = evaluate(opt, net, dataset['testset'])

    print ("\tepoch " .. epochCtr .. "/" .. opt.numEpochs .. ": trainLoss = " .. trainLoss .. ", trainAcc = " .. trainAcc .. ", valAcc = " .. valAcc .. ", testAcc = " .. testAcc)
end
--]]
