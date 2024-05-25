# independently trained model

# rgb+b1 band groups
python test.py --band-groups rgb --model-path output/RGB_LB/best_model.pth --output-dir output/RGB_LB  --test-folder test-90% # LB
python test.py --band-groups rgb --model-path output/RGB_UB/best_model.pth --output-dir output/RGB_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb --model-path output/RGB_ADG/best_model.pth --output-dir output/RGB_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
python test.py --band-groups rgb --model-path output/RGB_ADG_AUTO/best_model.pth --output-dir output/RGB_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb --model-path output/RGB_CORAL/best_model.pth --output-dir output/RGB_CORAL  --test-folder test-90% --dg coral # CORAL
python test.py --band-groups rgb --model-path output/RGB_CORAL_AUTO/best_model.pth --output-dir output/RGB_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)

#rgb+b1+nir band groups
python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB/best_model.pth --output-dir output/RGB+NIR_LB  --test-folder test-90% # LB
python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB/best_model.pth --output-dir output/RGB+NIR_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG/best_model.pth --output-dir output/RGB+NIR_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL/best_model.pth --output-dir output/RGB+NIR_CORAL  --test-folder test-90% --dg coral # CORAL
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)


# rgb+b1+nir+swir band groups
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB/best_model.pth --output-dir output/RGB+NIR+SWIR_LB  --test-folder test-90% # LB
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB/best_model.pth --output-dir output/RGB+NIR+SWIR_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL  --test-folder test-90% --dg coral # CORAL
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)

##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################

# pratrained models

#rgb+b1+nir band groups
python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR_LB_pretrained  --test-folder test-90% # LB
python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR_UB_pretrained  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_pretrained/best_model.pth --output-dir output/RGB+NIR_ADG_pretrained  --test-folder test-90% --dg adv # ADG (Adversarial DG)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_pretrained/best_model.pth --output-dir output/RGB+NIR_CORAL_pretrained  --test-folder test-90% --dg coral # CORAL
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)


# rgb+b1+nir+swir band groups
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_LB_pretrained  --test-folder test-90% # LB
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_UB_pretrained  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_pretrained  --test-folder test-90% --dg adv # ADG (Adversarial DG)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_pretrained  --test-folder test-90% --dg coral # CORAL
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
