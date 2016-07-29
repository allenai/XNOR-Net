--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local N
local top1Sum, top5Sum, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   N = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()
   
   if opt.binaryWeight then
      binarizeConvParms(convNodes)
   end
   
   top1Sum = 0
   top5Sum = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   if opt.binaryWeight then
      parameters:copy(realParams)
      if opt.nGPU >1 then
        model:syncParameters()
      end
   end

   loss = loss / N --(nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) '] = (top1Sum/N),
      ['% top5 accuracy (test set) '] = (top5Sum/N),
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1Sum/N, top5Sum/N))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------


local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
   batchNumber = batchNumber + opt.batchSize
   N = N + 1;

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local pred = outputs:float()

   local top1, top5 = computeScore(pred, labelsCPU, 1)
   top1Sum = top1Sum + top1
   top5Sum = top5Sum + top5
   
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d] | top1 : [%.2f (%.2f)] | top5 : [%.2f (%.2f)]'):format(epoch, batchNumber, nTest, top1 , top1Sum/N, top5, top5Sum/N))
   end
end
