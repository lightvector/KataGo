
---

Input Group:
conv_spatial ~ 3.3e-6
if 'conv_spatial' in name:
1.0: 



linear_global ~ 1.25e-5
if 'linear_global' in name:
1.0: 





Blocks Group:
'weight'
if 'blocks' in name and 'weight' in name:
1.0: 



Output Group:
not 'intermediate'
'policy_head'

'value_head'



Intermediate Group:
'intermediate_policy_head'

'intermediate_value_head'

if 'intermediate' in name:
    if 'policy_head' in name:
        continue
    elif 'value_head' in name:
        continue
else:
    if 'policy_head' in name:
        continue
    elif 'value_head' in name:
        continue

value:
1.0:



Normalization Group:
'norm.'(Blocks) ~ beta, gamma

'norm_trunkfinal'

if 'norm_trunkfinal' in name:
elif 'norm.' in name:
    if 'beta' in name:
    elif 'gamma' in name:

beta:
1.0:


gamma:
1.0:


norm_trunkfinal:
1.0:



lr-scale: 
1.0
abs_update_ratio:
conv_spatial abs_update_ratio: mean=9.424498112900133e-05, var=2.1925242023947163e-09, data length: 165
linear_global abs_update_ratio: mean=0.000102843158372836, var=4.981804472810054e-09, data length: 165
norm_beta abs_update_ratio: mean=1.089860709807587e-05, var=5.329024098036653e-10, data length: 28050
norm_gamma abs_update_ratio: mean=7.821163132267262e-06, var=1.5817057668903985e-10, data length: 27885
blocks abs_update_ratio: mean=6.106349898146968e-05, var=7.411311931407822e-10, data length: 30690
policy_head abs_update_ratio: mean=4.886833039850122e-06, var=4.097584112066781e-11, data length: 1485
value_head abs_update_ratio: mean=3.038486337519042e-05, var=7.361269325903134e-09, data length: 3630
intermediate_policy abs_update_ratio: mean=6.691634432025987e-06, var=4.385687730813803e-11, data length: 1485
intermediate_value abs_update_ratio: mean=1.969989352815589e-05, var=1.5308138645847073e-09, data length: 3630
update_ratio:
conv_spatial update_ratio: mean=7.406040365873041e-07, var=2.6009193542308592e-09, data length: 165
linear_global update_ratio: mean=2.887276998298657e-06, var=1.3908092110379902e-07, data length: 165
norm_beta update_ratio: mean=-8.851741257197293e-08, var=1.3509078195394503e-10, data length: 28050
norm_gamma update_ratio: mean=-3.069003398728121e-08, var=2.9401444281002104e-11, data length: 27885
blocks update_ratio: mean=0.0007289218770070653, var=0.014625711383350977, data length: 30690
policy_head update_ratio: mean=-1.4930936635322336e-07, var=3.884287206214445e-11, data length: 1485
value_head update_ratio: mean=-0.002541586221533767, var=0.1396044291466567, data length: 3630
intermediate_policy update_ratio: mean=-4.396897470896627e-07, var=3.086847757098784e-10, data length: 1485
intermediate_value update_ratio: mean=1.3391573491885158e-05, var=3.6940165112228052e-06, data length: 3630
weight:
conv_spatial weight: mean=-0.0037158203776925802, var=0.0027032638899981976, data length: 1
linear_global weight: mean=0.0008330182754434645, var=0.0543990284204483, data length: 1
norm_beta weight: mean=-0.7858779857263846, var=0.43572286571211677, data length: 170
norm_gamma weight: mean=1.162360256945593, var=0.10695687562403594, data length: 169
blocks weight: mean=-0.0012065561609380247, var=0.002470592488813269, data length: 186
policy_head weight: mean=-0.12854534809270668, var=0.13229456926799482, data length: 9
value_head weight: mean=0.006216140223180198, var=1.0859939510942653, data length: 22
intermediate_policy weight: mean=-0.30432747102652985, var=0.3913440385626422, data length: 9
intermediate_value weight: mean=-0.1127990368544984, var=2.222754522182268, data length: 22
gradient:
conv_spatial gradient: mean=-2.7296211721904206e-09, var=6.222348869057972e-11, data length: 165
linear_global gradient: mean=2.1247784716867575e-09, var=2.6061721784231493e-10, data length: 165
norm_beta gradient: mean=2.229302851382967e-09, var=2.110521249465591e-10, data length: 28050
norm_gamma gradient: mean=-3.671608394416115e-09, var=1.9811025819804103e-10, data length: 27885
blocks gradient: mean=-9.897146624248921e-10, var=1.982192158551915e-11, data length: 30690
policy_head gradient: mean=-1.4605192429401944e-08, var=3.68842357996779e-12, data length: 1485
value_head gradient: mean=6.68231831352432e-08, var=2.2476491968129155e-11, data length: 3630
intermediate_policy gradient: mean=-1.0945386865819915e-09, var=3.240532229432506e-11, data length: 1485
intermediate_value gradient: mean=2.430942336040095e-07, var=1.6195773145773949e-10, data length: 3630
pacc1: 0.577464


2.0
Absolute update ratio: 
conv_spatial Absolute update ratio: 0.00016084801294221553, data lenth: 2974
linear_global Absolute update ratio: 0.00016875826583404876, data lenth: 2974
norm_beta Absolute update ratio: 1.9184980759127863e-05, data lenth: 505580
norm_gamma Absolute update ratio: 1.4186502921068463e-05, data lenth: 502606
blocks Absolute update ratio: 0.0001115672066927935, data lenth: 553164
policy_head Absolute update ratio: 8.785183711202917e-06, data lenth: 26766
value_head Absolute update ratio: 6.151905382799498e-05, data lenth: 65428
intermediate_policy Absolute update ratio: 1.2367239983933635e-05, data lenth: 26766
intermediate_value Absolute update ratio: 3.713492134992417e-05, data lenth: 65428
Parameter update ratio: 
conv_spatial Parameter update ratio: 1.6490095276050857e-07, data lenth: 2974
linear_global Parameter update ratio: 3.5111654553769272e-06, data lenth: 2974
norm_beta Parameter update ratio: 1.7610938230062122e-09, data lenth: 505580
norm_gamma Parameter update ratio: 3.857837258915139e-09, data lenth: 502606
blocks Parameter update ratio: 3.222026902271935e-06, data lenth: 553164
policy_head Parameter update ratio: 6.3103534977405025e-09, data lenth: 26766
value_head Parameter update ratio: -0.004831001250440193, data lenth: 65428
intermediate_policy Parameter update ratio: -6.400716311570505e-08, data lenth: 26766
intermediate_value Parameter update ratio: 0.00018818180407983534, data lenth: 65428
pacc1: 0.586582
Absolute update ratio: 
conv_spatial Absolute update ratio: 0.00016083679446362198, data lenth: 2975
linear_global Absolute update ratio: 0.00016875210511388256, data lenth: 2975
norm_beta Absolute update ratio: 1.918413966111104e-05, data lenth: 505750
norm_gamma Absolute update ratio: 1.4185864455417714e-05, data lenth: 502775
blocks Absolute update ratio: 0.0001115629066675216, data lenth: 553350
policy_head Absolute update ratio: 8.785775712587775e-06, data lenth: 26775
value_head Absolute update ratio: 6.154499092685914e-05, data lenth: 65450
intermediate_policy Absolute update ratio: 1.2367923605394597e-05, data lenth: 26775
intermediate_value Absolute update ratio: 3.714195581520584e-05, data lenth: 65450
Parameter update ratio: 
conv_spatial Parameter update ratio: 1.571889710317045e-07, data lenth: 2975
linear_global Parameter update ratio: 3.61853462697532e-06, data lenth: 2975
norm_beta Parameter update ratio: 1.1189765652719868e-09, data lenth: 505750
norm_gamma Parameter update ratio: 3.7091505209217034e-09, data lenth: 502775
blocks Parameter update ratio: 3.225011916141909e-06, data lenth: 553350
policy_head Parameter update ratio: 4.013567372697247e-09, data lenth: 26775
value_head Parameter update ratio: -0.004828950498548238, data lenth: 65450
intermediate_policy Parameter update ratio: -6.28319471146674e-08, data lenth: 26775
intermediate_value Parameter update ratio: 0.0001880985401323719, data lenth: 65450
pacc1: 0.586655



---





Parameters in model:


conv_spatial.weight, [512, 22, 3, 3], 101376 params


linear_global.weight, [512, 19], 9728 params


blocks.0.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.0.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.0.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.0.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.0.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.0.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.0.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.0.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.0.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.0.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.0.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.0.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.0.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.0.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.0.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.0.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.0.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.0.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.1.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.1.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.1.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.1.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.1.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.1.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.1.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.1.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.1.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.1.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.1.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.1.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.1.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.1.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.1.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.1.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.1.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.1.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.2.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.2.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.2.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.2.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.2.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.2.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.2.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.2.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.2.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.2.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.2.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.2.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.2.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.2.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.2.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.2.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.2.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.2.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.2.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.2.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.2.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.2.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.3.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.3.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.3.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.3.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.3.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.3.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.3.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.3.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.3.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.3.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.3.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.3.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.3.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.3.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.3.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.3.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.3.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.3.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.4.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.4.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.4.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.4.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.4.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.4.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.4.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.4.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.4.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.4.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.4.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.4.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.4.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.4.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.4.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.4.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.4.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.4.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.5.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.5.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.5.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.5.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.5.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.5.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.5.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.5.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.5.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.5.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.5.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.5.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.5.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.5.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.5.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.5.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.5.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.5.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.5.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.5.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.5.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.5.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.6.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.6.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.6.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.6.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.6.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.6.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.6.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.6.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.6.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.6.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.6.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.6.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.6.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.6.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.6.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.6.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.6.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.6.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.7.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.7.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.7.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.7.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.7.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.7.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.7.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.7.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.7.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.7.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.7.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.7.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.7.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.7.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.7.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.7.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.7.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.7.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.8.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.8.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.8.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.8.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.8.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.8.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.8.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.8.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.8.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.8.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.8.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.8.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.8.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.8.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.8.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.8.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.8.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.8.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.8.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.8.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.8.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.8.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.9.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.9.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.9.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.9.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.9.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.9.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.9.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.9.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.9.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.9.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.9.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.9.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.9.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.9.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.9.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.9.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.9.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.9.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.10.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.10.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.10.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.10.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.10.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.10.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.10.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.10.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.10.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.10.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.10.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.10.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.10.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.10.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.10.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.10.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.10.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.10.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.11.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.11.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.11.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.11.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.11.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.11.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.11.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.11.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.11.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.11.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.11.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.11.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.11.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.11.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.11.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.11.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.11.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.11.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.11.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.11.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.11.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.11.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.12.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.12.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.12.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.12.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.12.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.12.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.12.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.12.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.12.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.12.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.12.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.12.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.12.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.12.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.12.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.12.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.12.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.12.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.13.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.13.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.13.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.13.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.13.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.13.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.13.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.13.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.13.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.13.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.13.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.13.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.13.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.13.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.13.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.13.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.13.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.13.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.14.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.14.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.14.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.14.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.14.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.14.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.14.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.14.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.14.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.14.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.14.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.14.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.14.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.14.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.14.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.14.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.14.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.14.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.14.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.14.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.14.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.14.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.15.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.15.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.15.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.15.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.15.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.15.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.15.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.15.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.15.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.15.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.15.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.15.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.15.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.15.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.15.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.15.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.15.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.15.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.16.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.16.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.16.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.16.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.16.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.16.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.16.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.16.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.16.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.16.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.16.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.16.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.16.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.16.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.16.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.16.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.16.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.16.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.17.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.17.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.17.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.17.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.17.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.17.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.17.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.17.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.17.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.17.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.17.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.17.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.17.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.17.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.17.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.17.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.17.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.17.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.17.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.17.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.17.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.17.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.18.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.18.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.18.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.18.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.18.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.18.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.18.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.18.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.18.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.18.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.18.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.18.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.18.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.18.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.18.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.18.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.18.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.18.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.19.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.19.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.19.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.19.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.19.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.19.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.19.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.19.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.19.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.19.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.19.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.19.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.19.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.19.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.19.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.19.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.19.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.19.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.20.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.20.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.20.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.20.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.20.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.20.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.20.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.20.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.20.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.20.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.20.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.20.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.20.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.20.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.20.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.20.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.20.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.20.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.20.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.20.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.20.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.20.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.21.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.21.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.21.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.21.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.21.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.21.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.21.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.21.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.21.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.21.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.21.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.21.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.21.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.21.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.21.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.21.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.21.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.21.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.22.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.22.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.22.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.22.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.22.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.22.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.22.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.22.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.22.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.22.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.22.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.22.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.22.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.22.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.22.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.22.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.22.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.22.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.23.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.23.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.23.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.23.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.23.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.23.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.23.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.23.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.23.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.23.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.23.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.23.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.23.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.23.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.23.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.23.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.23.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.23.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.23.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.23.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.23.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.23.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.24.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.24.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.24.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.24.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.24.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.24.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.24.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.24.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.24.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.24.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.24.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.24.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.24.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.24.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.24.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.24.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.24.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.24.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.25.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.25.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.25.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.25.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.25.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.25.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.25.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.25.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.25.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.25.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.25.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.25.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.25.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.25.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.25.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.25.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.25.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.25.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.26.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.26.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.26.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.26.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.26.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.26.blockstack.0.normactconv1.convpool.conv1r.weight, [192, 256, 3, 3], 442368 params
blocks.26.blockstack.0.normactconv1.convpool.conv1g.weight, [64, 256, 3, 3], 147456 params
blocks.26.blockstack.0.normactconv1.convpool.normg.beta, [1, 64, 1, 1], 64 params
blocks.26.blockstack.0.normactconv1.convpool.normg.gamma, [1, 64, 1, 1], 64 params
blocks.26.blockstack.0.normactconv1.convpool.linear_g.weight, [192, 192], 36864 params
blocks.26.blockstack.0.normactconv2.norm.beta, [1, 192, 1, 1], 192 params
blocks.26.blockstack.0.normactconv2.norm.gamma, [1, 192, 1, 1], 192 params
blocks.26.blockstack.0.normactconv2.conv.weight, [256, 192, 3, 3], 442368 params
blocks.26.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.26.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.26.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.26.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.26.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.26.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.26.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.26.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.26.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params
blocks.27.normactconvp.norm.beta, [1, 512, 1, 1], 512 params
blocks.27.normactconvp.norm.gamma, [1, 512, 1, 1], 512 params
blocks.27.normactconvp.conv.weight, [256, 512, 1, 1], 131072 params
blocks.27.blockstack.0.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.27.blockstack.0.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.27.blockstack.0.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.27.blockstack.0.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.27.blockstack.0.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.27.blockstack.0.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.27.blockstack.1.normactconv1.norm.beta, [1, 256, 1, 1], 256 params
blocks.27.blockstack.1.normactconv1.norm.gamma, [1, 256, 1, 1], 256 params
blocks.27.blockstack.1.normactconv1.conv.weight, [256, 256, 3, 3], 589824 params
blocks.27.blockstack.1.normactconv2.norm.beta, [1, 256, 1, 1], 256 params
blocks.27.blockstack.1.normactconv2.norm.gamma, [1, 256, 1, 1], 256 params
blocks.27.blockstack.1.normactconv2.conv.weight, [256, 256, 3, 3], 589824 params
blocks.27.normactconvq.norm.beta, [1, 256, 1, 1], 256 params
blocks.27.normactconvq.norm.gamma, [1, 256, 1, 1], 256 params
blocks.27.normactconvq.conv.weight, [512, 256, 1, 1], 131072 params


norm_trunkfinal.beta, [1, 512, 1, 1], 512 params


policy_head.conv1p.weight, [64, 512, 1, 1], 32768 params
policy_head.conv1g.weight, [64, 512, 1, 1], 32768 params
policy_head.biasg.beta, [1, 64, 1, 1], 64 params
policy_head.linear_g.weight, [64, 192], 12288 params
policy_head.linear_pass.weight, [64, 192], 12288 params
policy_head.linear_pass.bias, [64], 64 params
policy_head.linear_pass2.weight, [6, 64], 384 params
policy_head.bias2.beta, [1, 64, 1, 1], 64 params
policy_head.conv2p.weight, [6, 64, 1, 1], 384 params


value_head.conv1.weight, [128, 512, 1, 1], 65536 params
value_head.bias1.beta, [1, 128, 1, 1], 128 params
value_head.linear2.weight, [144, 384], 55296 params
value_head.linear2.bias, [144], 144 params
value_head.linear_valuehead.weight, [3, 144], 432 params
value_head.linear_valuehead.bias, [3], 3 params
value_head.linear_miscvaluehead.weight, [10, 144], 1440 params
value_head.linear_miscvaluehead.bias, [10], 10 params
value_head.linear_moremiscvaluehead.weight, [8, 144], 1152 params
value_head.linear_moremiscvaluehead.bias, [8], 8 params
value_head.conv_ownership.weight, [1, 128, 1, 1], 128 params
value_head.conv_scoring.weight, [1, 128, 1, 1], 128 params
value_head.conv_futurepos.weight, [2, 512, 1, 1], 1024 params
value_head.conv_seki.weight, [4, 512, 1, 1], 2048 params
value_head.linear_s2.weight, [128, 384], 49152 params
value_head.linear_s2.bias, [128], 128 params
value_head.linear_s2off.weight, [128, 1], 128 params
value_head.linear_s2par.weight, [128, 1], 128 params
value_head.linear_s3.weight, [8, 128], 1024 params
value_head.linear_s3.bias, [8], 8 params
value_head.linear_smix.weight, [8, 384], 3072 params
value_head.linear_smix.bias, [8], 8 params


norm_intermediate_trunkfinal.gamma, [1, 512, 1, 1], 512 params
norm_intermediate_trunkfinal.beta, [1, 512, 1, 1], 512 params


intermediate_policy_head.conv1p.weight, [64, 512, 1, 1], 32768 params
intermediate_policy_head.conv1g.weight, [64, 512, 1, 1], 32768 params
intermediate_policy_head.biasg.beta, [1, 64, 1, 1], 64 params
intermediate_policy_head.linear_g.weight, [64, 192], 12288 params
intermediate_policy_head.linear_pass.weight, [64, 192], 12288 params
intermediate_policy_head.linear_pass.bias, [64], 64 params
intermediate_policy_head.linear_pass2.weight, [6, 64], 384 params
intermediate_policy_head.bias2.beta, [1, 64, 1, 1], 64 params
intermediate_policy_head.conv2p.weight, [6, 64, 1, 1], 384 params


intermediate_value_head.conv1.weight, [128, 512, 1, 1], 65536 params
intermediate_value_head.bias1.beta, [1, 128, 1, 1], 128 params
intermediate_value_head.linear2.weight, [144, 384], 55296 params
intermediate_value_head.linear2.bias, [144], 144 params
intermediate_value_head.linear_valuehead.weight, [3, 144], 432 params
intermediate_value_head.linear_valuehead.bias, [3], 3 params
intermediate_value_head.linear_miscvaluehead.weight, [10, 144], 1440 params
intermediate_value_head.linear_miscvaluehead.bias, [10], 10 params
intermediate_value_head.linear_moremiscvaluehead.weight, [8, 144], 1152 params
intermediate_value_head.linear_moremiscvaluehead.bias, [8], 8 params
intermediate_value_head.conv_ownership.weight, [1, 128, 1, 1], 128 params
intermediate_value_head.conv_scoring.weight, [1, 128, 1, 1], 128 params
intermediate_value_head.conv_futurepos.weight, [2, 512, 1, 1], 1024 params
intermediate_value_head.conv_seki.weight, [4, 512, 1, 1], 2048 params
intermediate_value_head.linear_s2.weight, [128, 384], 49152 params
intermediate_value_head.linear_s2.bias, [128], 128 params
intermediate_value_head.linear_s2off.weight, [128, 1], 128 params
intermediate_value_head.linear_s2par.weight, [128, 1], 128 params
intermediate_value_head.linear_s3.weight, [8, 128], 1024 params
intermediate_value_head.linear_s3.bias, [8], 8 params
intermediate_value_head.linear_smix.weight, [8, 384], 3072 params
intermediate_value_head.linear_smix.bias, [8], 8 params

Total num params: 73162378
Total trainable params: 73162378












