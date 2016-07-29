--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
local ffi=require 'ffi'

function computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   local top1 = correct:narrow(2, 1, 1):sum() / batchSize
   local top5 = correct:narrow(2, 1, 5):sum() / batchSize

   return top1 * 100, top5 * 100
end

function makeDataParallel(model, nGPU)

   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
   end
   cutorch.setDevice(opt.GPU)

   return model
end

local function cleanDPT(module)
   return module:get(1)
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      torch.save(filename, model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadParams(model,saved_model)
    params = model:parameters();
    local saved_params = saved_model:parameters(); 
         for i=1,#params do
            params[i]:copy(saved_params[i]);
         end
      local bn= model:findModules("nn.SpatialBatchNormalization")
      local saved_bn= saved_model:findModules("nn.SpatialBatchNormalization")
      for i=1,#bn do
         bn[i].running_mean:copy(saved_bn[i].running_mean)
         bn[i].running_var:copy(saved_bn[i].running_var)
      end
end


function zeroBias(convNodes)
   for i =1, #convNodes do
    local n = convNodes[i].bias:fill(0)
   end
end
function updateBinaryGradWeight(convNodes)
   for i =2, #convNodes-1 do
    local n = convNodes[i].weight[1]:nElement()
    local s = convNodes[i].weight:size()
    local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n):expand(s);
    m[convNodes[i].weight:le(-1)]=0;
    m[convNodes[i].weight:ge(1)]=0;
    m:add(1/(n)):mul(1-1/s[2]):mul(n);
    convNodes[i].gradWeight:cmul(m)--:cmul(mg)
   end
   if opt.nGPU >1 then
    model:syncParameters()
   end
end




function meancenterConvParms(convNodes)
   for i =2, #convNodes-1 do
    local s = convNodes[i].weight:size()
    local negMean = convNodes[i].weight:mean(2):mul(-1):repeatTensor(1,s[2],1,1);  
    convNodes[i].weight:add(negMean)
   end
   if opt.nGPU >1 then
    model:syncParameters()
   end
end


function binarizeConvParms(convNodes)
   for i =2, #convNodes-1 do
    local n = convNodes[i].weight[1]:nElement()
    local s = convNodes[i].weight:size()
    
    local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
    convNodes[i].weight:sign():cmul(m:expand(s))
   end
   if opt.nGPU >1 then
    model:syncParameters()
   end
end


function clampConvParms(convNodes)
   for i =2, #convNodes-1 do
    convNodes[i].weight:clamp(-1,1)
   end
   if opt.nGPU >1 then
    model:syncParameters()
   end
end



function rand_initialize(layer)
  local tn = torch.type(layer)
  if tn == "cudnn.SpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.SpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.BinarySpatialConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.SpatialConvolutionMM" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "cudnn.VolumetricConvolution" then
    local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.Linear" then
    local c =  math.sqrt(2.0 / layer.weight:size(2));
    layer.weight:copy(torch.randn(layer.weight:size()) * c)
    layer.bias:fill(0)
  elseif tn == "nn.SpatialBachNormalization" then
    layer.weight:fill(1)
    layer.bias:fill(0)
  elseif tn == "cudnn.SpatialBachNormalization" then
    layer.weight:fill(1)
    layer.bias:fill(0)
  end
end
