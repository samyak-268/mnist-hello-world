function evaluate(opt, net, dataset)
    net:evaluate()

    local count = 0.
    local batchSize = opt.batchSize
    local iterCnt, valLoss = 0, 0
    for startIdx = 1, dataset.size, opt.batchSize do
        local endIdx = math.min(startIdx + batchSize - 1, dataset.size)
        local size = (endIdx - startIdx + 1)

        local batchInput, batchLabel = torch.Tensor(size, 28, 28), torch.Tensor(size)
        for offset = 0, (size-1) do
            local image = dataset.data[startIdx + offset]:add(-opt.trainingMean)
            local label = dataset.label[startIdx + offset] + 1
            batchInput[offset+1], batchLabel[offset+1] = image, label
        end

        if opt.gpu == 1 then
            batchInput = batchInput:cuda()
            batchLabel = batchLabel:cuda()
            opt.criterion = opt.criterion:cuda()
        end

        local batchOutput = net:forward(batchInput)
        local _, maxIndexes = torch.max(batchOutput, 2)
        valLoss = valLoss + opt.criterion:forward(batchOutput, batchLabel)

        if opt.gpu == 1 then
            rightGuesses = maxIndexes:eq(batchLabel:cudaLong()):sum()
        else
            rightGuesses = maxIndexes:eq(batchLabel:long()):sum()
        end

        count = count + rightGuesses
        iterCnt = (iterCnt + 1)
    end

    net:training()
    return valLoss / iterCnt, count / dataset.size 
end
