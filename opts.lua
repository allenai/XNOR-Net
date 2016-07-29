--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', './cache/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './imagenet/imagenet_raw_images/256', 'Home of ImageNet dataset')
    cmd:option('-dataset',  'imagenet', 'Dataset Name: imagenet |cifar')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | ccn2 | cunn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        1000, 'number of classes in the dataset')
    cmd:option('-scalingFactor',   0, 'number of classes in the dataset')

    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       2500, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     0, 'weight decay')
    cmd:option('-shareGradInput',  true, 'Sharing the gradient memory')
    cmd:option('-binaryWeight',    false, 'Sharing the gradient memory')
    cmd:option('-testOnly',    false, 'Sharing the gradient memory')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'alexnet', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet | resnet')
    cmd:option('-optimType',     'sgd', 'Options: sgd | adam')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-loadParams',  'none', 'provide path to model to load the parameters')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-depth',  18, 'Depth for resnet')
    cmd:option('-shortcutType',  'B', 'type of short cut in resnet: A|B|C')
    cmd:option('-dropout', 0.5 , 'Dropout ratio')

    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string(opt.netType, opt,
                                       {netType=true, retrain=true, loadParams=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
