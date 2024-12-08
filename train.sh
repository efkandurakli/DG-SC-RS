# train each band-group independently
# (rgb+b1) band groups
python train.py  --output-dir output/RGB_LB --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB_UB --band-groups rgb  --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB_ADG_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB_CORAL_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150 --lr 0.01  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
python train_mixstyle.py  --output-dir output/RGB_MIXSTYLE --band-groups rgb --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
python train_irm.py  --output-dir output/RGB_IRM_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150 --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)

# (rgb+b1)+nir band groups
python train.py  --output-dir output/RGB+NIR_LB --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR_UB --band-groups rgb nir --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
python train_mixstyle.py  --output-dir output/RGB+NIR_MIXSTYLE --band-groups rgb nir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
python train_irm.py  --output-dir output/RGB+NIR_IRM_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)

# (rgb+b1)+nir+swir band groups
python train.py  --output-dir output/RGB+NIR+SWIR_LB --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR+SWIR_UB --band-groups rgb nir swir --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
python train_mixstyle.py  --output-dir output/RGB+NIR+SWIR_MIXSTYLE --band-groups rgb nir swir --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)


###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################

# train using previously trained model

# (rgb+b1)+nir band groups
python train.py  --output-dir output/RGB+NIR_LB_pretrained --pretrained-model output/RGB_LB/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR_UB_pretrained --pretrained-model output/RGB_UB/best_model.pth --band-groups rgb nir --num-channels 4 --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO_pretrained --pretrained-model output/RGB_ADG_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO_pretrained --pretrained-model output/RGB_CORAL_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
python train_mixstyle.py  --output-dir output/RGB+NIR_MIXSTYLE_pretrained --pretrained-model output/RGB_MIXSTYLE/best_model.pth --band-groups rgb nir --num-channels 4 --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
python train_irm.py  --output-dir output/RGB+NIR_IRM_AUTO_pretrained --pretrained-model output/RGB_IRM_AUTO/best_model.pth --band-groups rgb nir --num-channels 4 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)

# (rgb+b1)+nir+swir band groups
python train.py  --output-dir output/RGB+NIR+SWIR_LB_pretrained --pretrained-model output/RGB+NIR_LB_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR+SWIR_UB_pretrained --pretrained-model output/RGB+NIR_UB_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --train-folders train test-10% --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained --pretrained-model output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained --pretrained-model output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
python train_mixstyle.py  --output-dir output/RGB+NIR+SWIR_MIXSTYLE_pretrained --pretrained-model output/RGB+NIR_MIXSTYLE_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9  --mix-layers layer1 layer2 layer3 --train-folders train --use-deterministic-algorithms --patience 150  --max-epochs 150 --batch-size 128 # MIXSTYLE
python train_irm.py  --output-dir output/RGB+NIR+SWIR_IRM_AUTO_pretrained --pretrained-model output/RGB+NIR_IRM_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --num-channels 9 --auto-weighted-loss --train-folders train --use-deterministic-algorithms --lr 0.01 --patience 150  --max-epochs 150 --batch-size 129 # IRM_AUTO (auto weighted loss)
