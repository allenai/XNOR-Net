--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },


    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1Sum, top5Sum, loss_epoch





-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState.learningRate = params.learningRate
      optimState.learningRateDecay = 0.0
      optimState.momentum = opt.momentum
      optimState.dampening = 0.0
      optimState.weightDecay = params.weightDecay
      if (opt.optimType == "adam") or (opt.optimType == "adamax") then
        optimState.learningRate =optimState.learningRate*0.1
        --optimState.t =  1
      end
   end
   

   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1Sum = 0
   top5Sum = 0
   loss_epoch = 0 
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1Sum/opt.epochSize,
      ['% top5 accuracy (train set)'] = top5Sum/opt.epochSize,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1Sum/opt.epochSize))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   --model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
collectgarbage()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()
local procTimer = torch.Timer()


-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   procTimer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

    local err, outputs
    if opt.binaryWeight then
     meancenterConvParms(convNodes)
     clampConvParms(convNodes)
     realParams:copy(parameters)
     binarizeConvParms(convNodes)
    end
    
    model:zeroGradParameters()
    outputs = model:forward(inputs)
    err = criterion:forward(outputs, labels)
    
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(inputs, gradOutputs)      
    
    if opt.binaryWeight then
      parameters:copy(realParams)
      updateBinaryGradWeight(convNodes)
      if opt.optimType == 'adam' then
        gradParameters:mul(1e+9);
      end
      if opt.nGPU >1 then
        model:syncParameters()
      end
    end

  local feval = function()
      return err, gradParameters
   end
   
   if opt.optimType == "sgd" then
      optim.sgd(feval, parameters, optimState)
   elseif opt.optimType == "adam" then
      optim.adam(feval, parameters, optimState)
   elseif opt.optimType == "adamax" then
      optim.adam(feval, parameters, optimState)
   end
   
   


   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end
   

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   

   local pred = outputs:float()
     local top1, top5 = computeScore(pred, labelsCPU, 1)
     top1Sum = top1Sum + top1
     top5Sum = top5Sum + top5
     -- Calculate top-1 error, and print information
     print(('Epoch: [%d][%d/%d]\tTime %.3f(%.3f) Err %.4f Top1-%%: %.2f (%.2f)  Top5-%%: %.2f (%.2f) LR %.0e DataTime %.3f'):format(
            epoch, batchNumber, opt.epochSize, timer:time().real ,procTimer:time().real ,err, top1, top1Sum/batchNumber, top5, top5Sum/batchNumber, optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
   timer:reset()
end
