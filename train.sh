# train each band-group independently
# rgb+b1 band groups
python train.py  --output-dir output/RGB_LB --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB_UB --band-groups rgb  --train-folders train test-10% --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB_ADG --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
python train_adv.py  --output-dir output/RGB_ADG_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB_CORAL --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL
python train_coral.py  --output-dir output/RGB_CORAL_AUTO --band-groups rgb --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)

#rgb+b1+nir band groups
python train.py  --output-dir output/RGB+NIR_LB --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR_UB --band-groups rgb nir --train-folders train test-10% --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR_ADG --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR_CORAL --band-groups rgb  nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL
python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)

#rgb+b1+nir+swir band groups
python train.py  --output-dir output/RGB+NIR+SWIR_LB --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR+SWIR_UB --band-groups rgb nir swir --train-folders train test-10% --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG (Adversarial DG)
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL --band-groups rgb  nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)

##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################

# train using previously trained model

#rgb+b1+nir band groups
python train.py  --output-dir output/RGB+NIR_LB_pretrained --pretrained-model output/RGB_LB/best_model.pth --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR_UB_pretrained --pretrained-model output/RGB_UB/best_model.pth --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR_ADG_pretrained --pretrained-model output/RGB_ADG/best_model.pth --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG
python train_adv.py  --output-dir output/RGB+NIR_ADG_AUTO_pretrained --pretrained-model output/RGB_ADG_AUTO/best_model.pth --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR_CORAL_pretrained --pretrained-model output/RGB_CORAL/best_model.pth --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129  # CORAL
python train_coral.py  --output-dir output/RGB+NIR_CORAL_AUTO_pretrained --pretrained-model output/RGB_CORAL_AUTO/best_model.pth --band-groups rgb nir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)

#rgb+b1+nir+swir band groups
python train.py  --output-dir output/RGB+NIR+SWIR_LB_pretrained --pretrained-model output/RGB+NIR_LB_pretrained/best_model.pth --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # LB
python train.py  --output-dir output/RGB+NIR+SWIR_UB_pretrained --pretrained-model output/RGB+NIR_UB_pretrained/best_model.pth --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # UB
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_pretrained --pretrained-model output/RGB+NIR_ADG_pretrained/best_model.pth --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 128 # ADG
python train_adv.py  --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained --pretrained-model output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # ADG_AUTO (auto weighted loss)
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_pretrained --pretrained-model output/RGB+NIR_CORAL_pretrained/best_model.pth --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129  # CORAL
python train_coral.py  --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained --pretrained-model output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --band-groups rgb nir swir --auto-weighted-loss --train-folders train --use-deterministic-algorithms --patience 150  ---max-epochs 150 --batch-size 129 # CORAL_AUTO (auto weighted loss)
