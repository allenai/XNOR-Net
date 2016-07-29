--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain) -- defined in util.lua
   model:clearState()
else
   paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
   --Initializing the parameters 
   model:apply(rand_initialize)
   if opt.loadParams ~= 'none' then
      local saved_model = torch.load(opt.loadParams)
      --local saved_params = torch.load(opt.loadParams)
      loadParams(model,saved_model);
   end
end
--debugger.enter()
-- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
-- containers override backwards to call backwards recursively on submodules
if opt.shareGradInput then
   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         if cache[moduleType] == nil then
            cache[moduleType] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[moduleType], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end


--  apply parallel 
model:cuda()
if opt.nGPU >1 then
   model = makeDataParallel(model, opt.nGPU,true,true)
end

--getting the parameters and gradient pointers
parameters, gradParameters = model:getParameters()
realParams = parameters:clone()
convNodes = model:findModules('cudnn.SpatialConvolution')



-- much faster
 cudnn.fastest = true
 cudnn.benchmark = true
-- 2. Create Criterion
criterion = nn.ClassNLLCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()

collectgarbage()
