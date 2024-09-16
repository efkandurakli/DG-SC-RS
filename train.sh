## train each band-group independently
## rgb+b1 band groups
#python train.py  --output-dir output/RGB_LB --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output/RGB_UB --band-groups rgb  --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
#python train_adv.py  --output-dir output/RGB_ADG --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
#python train_adv.py  --output-dir output/RGB_ADG_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
#python train_coral.py  --output-dir output/RGB_CORAL --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # CORAL
#python train_coral.py  --output-dir output/RGB_CORAL_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150 --lr 0.01  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
#python train_mixstyle.py  --output-dir output/RGB_MIXSTYLE --band-groups rgb --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_irm.py  --output-dir output/RGB_IRM --band-groups rgb  --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM
#python train_irm.py  --output-dir output/RGB_IRM_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150 --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)


#rgb+b1+nir band groups
#python train.py  --output-dir output/RGB+NIR_LB --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output/RGB+NIR_UB --band-groups rgb nir --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
#python train_adv.py  --output-dir output/RGB+NIR_ADG --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
#python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
#python train_coral.py  --output-dir output/RGB+NIR_CORAL --band-groups rgb  nir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # CORAL
#python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
#python train_mixstyle.py  --output-dir output/RGB+NIR_MIXSTYLE --band-groups rgb nir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_irm.py  --output-dir output/RGB+NIR_IRM --band-groups rgb  nir --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM
#python train_irm.py  --output-dir output/RGB+NIR_IRM_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)

##rgb+b1+nir+swir band groups
#python train.py  --output-dir output/RGB+NIR+SWIR_LB --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output/RGB+NIR+SWIR_UB --band-groups rgb nir swir --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
#python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
#python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
#python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL --band-groups rgb  nir swir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # CORAL
#python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
#python train_mixstyle.py  --output-dir output/RGB+NIR+SWIR_MIXSTYLE --band-groups rgb nir swir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM --band-groups rgb  nir swir --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM
#python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)


###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################

# train using previously trained model

#rgb+b1+nir band groups
#python train.py  --output-dir output/RGB+NIR_LB_pretrained --pretrained-model output/RGB_LB/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output/RGB+NIR_UB_pretrained --pretrained-model output/RGB_UB/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
#python train_adv.py  --output-dir output/RGB+NIR_ADG_pretrained --pretrained-model output/RGB_ADG/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG
#python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO_pretrained --pretrained-model output/RGB_ADG_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
#python train_coral.py  --output-dir output/RGB+NIR_CORAL_pretrained --pretrained-model output/RGB_CORAL/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129  # CORAL
#python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO_pretrained --pretrained-model output/RGB_CORAL_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
#python train_mixstyle.py  --output-dir output/RGB+NIR_MIXSTYLE_pretrained --pretrained-model output/RGB_MIXSTYLE/best_model.pth --band-groups rgb nir --num-channels 4 --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_irm.py  --output-dir output/RGB+NIR_IRM_pretrained --pretrained-model output/RGB_IRM/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129  # IRM
#python train_irm.py  --output-dir output/RGB+NIR_IRM_AUTO_pretrained --pretrained-model output/RGB_IRM_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)

#rgb+b1+nir+swir band groups
#python train.py  --output-dir output/RGB+NIR+SWIR_LB_pretrained --pretrained-model output/RGB+NIR_LB_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output/RGB+NIR+SWIR_UB_pretrained --pretrained-model output/RGB+NIR_UB_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
#python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_pretrained --pretrained-model output/RGB+NIR_ADG_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG
#python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained --pretrained-model output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
#python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_pretrained --pretrained-model output/RGB+NIR_CORAL_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129  # CORAL
#python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained --pretrained-model output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
#python train_mixstyle.py  --output-dir output/RGB+NIR+SWIR_MIXSTYLE_pretrained --pretrained-model output/RGB+NIR_MIXSTYLE_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9  --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM_pretrained --pretrained-model output/RGB+NIR_IRM_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129  # IRM
#python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM_AUTO_pretrained --pretrained-model output/RGB+NIR_IRM_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)


#####  Ablation Study (Order of band groups)

# rgb+b1 band groups
#python train.py  --output-dir output2/RGB_LB --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/NIR_LB --band-groups nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR_LB --band-groups swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB

#python train_mixstyle.py  --output-dir output2/RGB_MIXSTYLE --band-groups rgb --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/NIR_MIXSTYLE --band-groups nir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR_MIXSTYLE --band-groups swir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE

#python train.py  --output-dir output2/NIR+SWIR_LB --band-groups nir swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR+RGB_LB --band-groups swir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB


#python train_mixstyle.py  --output-dir output2/NIR+SWIR_MIXSTYLE --band-groups nir swir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR+RGB_MIXSTYLE --band-groups swir rgb --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE

#python train.py  --output-dir output2/RGB+SWIR_LB_pretrained --pretrained-model output/RGB_LB/best_model.pth --num-channels 4 --band-groups rgb swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/NIR+RGB_LB_pretrained --pretrained-model output2/NIR_LB/best_model.pth --num-channels 5 --band-groups nir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR+RGB_LB_pretrained --pretrained-model output2/SWIR_LB/best_model.pth --num-channels 3 --band-groups swir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/NIR+SWIR_LB_pretrained --pretrained-model output2/NIR_LB/best_model.pth --num-channels 5 --band-groups nir swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR+NIR_LB_pretrained --pretrained-model output2/SWIR_LB/best_model.pth --num-channels 3 --band-groups swir nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128  # LB

#python train_mixstyle.py  --output-dir output2/RGB+SWIR_MIXSTYLE_pretrained --pretrained-model output/RGB_MIXSTYLE/best_model.pth --num-channels 4 --band-groups rgb swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/NIR+RGB_MIXSTYLE_pretrained --pretrained-model output2/NIR_MIXSTYLE/best_model.pth --num-channels 5 --band-groups nir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR+RGB_MIXSTYLE_pretrained --pretrained-model output2/SWIR_MIXSTYLE/best_model.pth --num-channels 3 --band-groups swir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/NIR+SWIR_MIXSTYLE_pretrained --pretrained-model output2/NIR_MIXSTYLE/best_model.pth --num-channels 5 --band-groups nir swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR+NIR_MIXSTYLE_pretrained --pretrained-model output2/SWIR_MIXSTYLE/best_model.pth --num-channels 3 --band-groups swir nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128  # MIXSTYLE

#python train.py  --output-dir output2/RGB+SWIR+NIR_LB_pretrained --pretrained-model output2/RGB+SWIR_LB_pretrained/best_model.pth --num-channels 7 --band-groups rgb swir nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/NIR+RGB+SWIR_LB_pretrained --pretrained-model output2/NIR+RGB_LB_pretrained/best_model.pth --num-channels 9 --band-groups nir rgb swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR+RGB+NIR_LB_pretrained --pretrained-model output2/SWIR+RGB_LB_pretrained/best_model.pth --num-channels 7 --band-groups swir rgb nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/NIR+SWIR+RGB_LB_pretrained --pretrained-model output2/NIR+SWIR_LB_pretrained/best_model.pth --num-channels 8 --band-groups nir swir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
#python train.py  --output-dir output2/SWIR+NIR+RGB_LB_pretrained --pretrained-model output2/SWIR+NIR_LB_pretrained/best_model.pth --num-channels 8 --band-groups swir nir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128  # LB


#python train_mixstyle.py  --output-dir output2/RGB+SWIR+NIR_MIXSTYLE_pretrained --pretrained-model output2/RGB+SWIR_MIXSTYLE_pretrained/best_model.pth --num-channels 7 --band-groups rgb swir nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/NIR+RGB+SWIR_MIXSTYLE_pretrained --pretrained-model output2/NIR+RGB_MIXSTYLE_pretrained/best_model.pth --num-channels 9 --band-groups nir rgb swir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR+RGB+NIR_MIXSTYLE_pretrained --pretrained-model output2/SWIR+RGB_MIXSTYLE_pretrained/best_model.pth --num-channels 7 --band-groups swir rgb nir  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/NIR+SWIR+RGB_MIXSTYLE_pretrained --pretrained-model output2/NIR+SWIR_MIXSTYLE_pretrained/best_model.pth --num-channels 8 --band-groups nir swir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
#python train_mixstyle.py  --output-dir output2/SWIR+NIR+RGB_MIXSTYLE_pretrained --pretrained-model output2/SWIR+NIR_MIXSTYLE_pretrained/best_model.pth --num-channels 8 --band-groups swir nir rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128  # MIXSTYLE

####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################

#python train_mixstyle.py  --output-dir output3/1 --band-groups b01  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/2 --band-groups b01 b02  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/3 --band-groups b01 b02 b03  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/4 --band-groups b01 b02 b03 b04  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/5 --band-groups b01 b02 b03 b04 b05  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/6 --band-groups b01 b02 b03 b04 b05 b06  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/7 --band-groups b01 b02 b03 b04 b05 b06 b07  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/8 --band-groups b01 b02 b03 b04 b05 b06 b07 b08  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/9 --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/10 --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/11 --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --output-dir output3/12 --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 b12  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128

#python train_mixstyle.py  --pretrained-model output3/1/best_model.pth --num-channels 1 --output-dir output3/2_pretrained --band-groups b01 b02  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/2_pretrained/best_model.pth --num-channels 2 --output-dir output3/3_pretrained --band-groups b01 b02 b03  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/3_pretrained/best_model.pth --num-channels 3 --output-dir output3/4_pretrained --band-groups b01 b02 b03 b04  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/4_pretrained/best_model.pth --num-channels 4 --output-dir output3/5_pretrained --band-groups b01 b02 b03 b04 b05  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/5_pretrained/best_model.pth --num-channels 5 --output-dir output3/6_pretrained --band-groups b01 b02 b03 b04 b05 b06  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/6_pretrained/best_model.pth --num-channels 6 --output-dir output3/7_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
#python train_mixstyle.py  --pretrained-model output3/7_pretrained/best_model.pth --num-channels 7 --output-dir output3/8_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07 b08  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
python train_mixstyle.py  --pretrained-model output3/8_pretrained/best_model.pth --num-channels 8 --output-dir output3/9_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
python train_mixstyle.py  --pretrained-model output3/9_pretrained/best_model.pth --num-channels 9 --output-dir output3/10_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
python train_mixstyle.py  --pretrained-model output3/10_pretrained/best_model.pth --num-channels 10 --output-dir output3/11_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128
python train_mixstyle.py  --pretrained-model output3/11_pretrained/best_model.pth --num-channels 11 --output-dir output3/12_pretrained --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 b12  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128