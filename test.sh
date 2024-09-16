# independently trained model
# rgb+b1 band groups
#python test.py --band-groups rgb --model-path output/RGB_LB/best_model.pth --output-dir output/RGB_LB  --test-folder test-90% # LB
#python test.py --band-groups rgb --model-path output/RGB_UB/best_model.pth --output-dir output/RGB_UB  --test-folder test-90% # upper bound
#python test.py --band-groups rgb --model-path output/RGB_ADG/best_model.pth --output-dir output/RGB_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
#python test.py --band-groups rgb --model-path output/RGB_ADG_AUTO/best_model.pth --output-dir output/RGB_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
#python test.py --band-groups rgb --model-path output/RGB_CORAL/best_model.pth --output-dir output/RGB_CORAL  --test-folder test-90% --dg coral # CORAL
#python test.py --band-groups rgb --model-path output/RGB_CORAL_AUTO/best_model.pth --output-dir output/RGB_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
#python test.py --band-groups rgb --model-path output/RGB_MIXSTYLE/best_model.pth --output-dir output/RGB_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups rgb --model-path output/RGB_IRM/best_model.pth --output-dir output/RGB_IRM  --test-folder test-90% # IRM
#python test.py --band-groups rgb --model-path output/RGB_IRM_AUTO/best_model.pth --output-dir output/RGB_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)

#rgb+b1+nir band groups
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB/best_model.pth --output-dir output/RGB+NIR_LB  --test-folder test-90% # LB
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB/best_model.pth --output-dir output/RGB+NIR_UB  --test-folder test-90% # upper bound
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG/best_model.pth --output-dir output/RGB+NIR_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL/best_model.pth --output-dir output/RGB+NIR_CORAL  --test-folder test-90% --dg coral # CORAL
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_MIXSTYLE/best_model.pth --output-dir output/RGB+NIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM/best_model.pth --output-dir output/RGB+NIR_IRM  --test-folder test-90% # IRM
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM_AUTO/best_model.pth --output-dir output/RGB+NIR_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)


# rgb+b1+nir+swir band groups
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB/best_model.pth --output-dir output/RGB+NIR+SWIR_LB  --test-folder test-90% # LB
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB/best_model.pth --output-dir output/RGB+NIR+SWIR_UB  --test-folder test-90% # upper bound
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG  --test-folder test-90% --dg adv # ADG (Adversarial DG)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL  --test-folder test-90% --dg coral # CORAL
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_MIXSTYLE/best_model.pth --output-dir output/RGB+NIR+SWIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM  --test-folder test-90% # IRM
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)

###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
## pratrained models

#rgb+b1+nir band groups
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR_LB_pretrained  --test-folder test-90% # LB
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR_UB_pretrained  --test-folder test-90% # upper bound
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_pretrained/best_model.pth --output-dir output/RGB+NIR_ADG_pretrained  --test-folder test-90% --dg adv # ADG (Adversarial DG)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_pretrained/best_model.pth --output-dir output/RGB+NIR_CORAL_pretrained  --test-folder test-90% --dg coral # CORAL
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_MIXSTYLE_pretrained/best_model.pth --output-dir output/RGB+NIR_MIXSTYLE_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM_pretrained/best_model.pth --output-dir output/RGB+NIR_IRM_pretrained  --test-folder test-90% # IRM
#python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_IRM_AUTO_pretrained  --test-folder test-90% # IRM_AUTO (auto weighted loss)


# rgb+b1+nir+swir band groups
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_LB_pretrained  --test-folder test-90% # LB
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_UB_pretrained  --test-folder test-90% # upper bound
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_pretrained  --test-folder test-90% --dg adv # ADG (Adversarial DG)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_pretrained  --test-folder test-90% --dg coral # CORAL
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_MIXSTYLE_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_MIXSTYLE_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM_pretrained  --test-folder test-90% # IRM
#python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM_AUTO_pretrained  --test-folder test-90% # IRM_AUTO (auto weighted loss)


#####  Ablation Study (Order of band groups)


#python test.py --band-groups rgb --model-path output2/RGB_LB/best_model.pth --output-dir output2/RGB_LB  --test-folder test-90% # LB
#python test.py --band-groups nir --model-path output2/NIR_LB/best_model.pth --output-dir output2/NIR_LB  --test-folder test-90% # LB
#python test.py --band-groups swir --model-path output2/SWIR_LB/best_model.pth --output-dir output2/SWIR_LB  --test-folder test-90% # LB

#python test.py --band-groups nir --model-path output2/NIR_MIXSTYLE/best_model.pth --output-dir output2/NIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir --model-path output2/SWIR_MIXSTYLE/best_model.pth --output-dir output2/SWIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE

#python test.py --band-groups nir swir --model-path output2/NIR+SWIR_LB/best_model.pth --output-dir output2/NIR+SWIR_LB  --test-folder test-90% # LB
#python test.py --band-groups swir rgb --model-path output2/SWIR+RGB_LB/best_model.pth --output-dir output2/SWIR+RGB_LB  --test-folder test-90% # LB

#python test.py --band-groups nir swir --model-path output2/NIR+SWIR_MIXSTYLE/best_model.pth --output-dir output2/NIR+SWIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir rgb --model-path output2/SWIR+RGB_MIXSTYLE/best_model.pth --output-dir output2/SWIR+RGB_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE

#python test.py --band-groups rgb swir --model-path output2/RGB+SWIR_LB_pretrained/best_model.pth --output-dir output2/RGB+SWIR_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups nir rgb --model-path output2/NIR+RGB_LB_pretrained/best_model.pth --output-dir output2/NIR+RGB_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups swir rgb --model-path output2/SWIR+RGB_LB_pretrained/best_model.pth --output-dir output2/SWIR+RGB_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups nir swir --model-path output2/NIR+SWIR_LB_pretrained/best_model.pth --output-dir output2/NIR+SWIR_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups swir nir --model-path output2/SWIR+NIR_LB_pretrained/best_model.pth --output-dir output2/SWIR+NIR_LB_pretrained --test-folder test-90% # LB

#python test.py --band-groups rgb swir --model-path output2/RGB+SWIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/RGB+SWIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups nir rgb --model-path output2/NIR+RGB_MIXSTYLE_pretrained/best_model.pth --output-dir output2/NIR+RGB_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir rgb --model-path output2/SWIR+RGB_MIXSTYLE_pretrained/best_model.pth --output-dir output2/SWIR+RGB_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups nir swir --model-path output2/NIR+SWIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/NIR+SWIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir nir --model-path output2/SWIR+NIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/SWIR+NIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE

#python test.py --band-groups rgb swir nir --model-path output2/RGB+SWIR+NIR_LB_pretrained/best_model.pth --output-dir output2/RGB+SWIR+NIR_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups nir rgb swir --model-path output2/NIR+RGB+SWIR_LB_pretrained/best_model.pth --output-dir output2/NIR+RGB+SWIR_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups swir rgb nir --model-path output2/SWIR+RGB+NIR_LB_pretrained/best_model.pth --output-dir output2/SWIR+RGB+NIR_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups nir swir rgb --model-path output2/NIR+SWIR+RGB_LB_pretrained/best_model.pth --output-dir output2/NIR+SWIR+RGB_LB_pretrained --test-folder test-90% # LB
#python test.py --band-groups swir nir rgb --model-path output2/SWIR+NIR+RGB_LB_pretrained/best_model.pth --output-dir output2/SWIR+NIR+RGB_LB_pretrained --test-folder test-90% # LB

#python test.py --band-groups rgb swir nir --model-path output2/RGB+SWIR+NIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/RGB+SWIR+NIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups nir rgb swir --model-path output2/NIR+RGB+SWIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/NIR+RGB+SWIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir rgb nir --model-path output2/SWIR+RGB+NIR_MIXSTYLE_pretrained/best_model.pth --output-dir output2/SWIR+RGB+NIR_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups nir swir rgb --model-path output2/NIR+SWIR+RGB_MIXSTYLE_pretrained/best_model.pth --output-dir output2/NIR+SWIR+RGB_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups swir nir rgb --model-path output2/SWIR+NIR+RGB_MIXSTYLE_pretrained/best_model.pth --output-dir output2/SWIR+NIR+RGB_MIXSTYLE_pretrained --test-folder test-90% --dg mixstyle # MIXSTYLE

####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################

#python test.py --band-groups b01 --model-path output3/1/best_model.pth --output-dir output3/1  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 --model-path output3/2/best_model.pth --output-dir output3/2  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 --model-path output3/3/best_model.pth --output-dir output3/3  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 --model-path output3/4/best_model.pth --output-dir output3/4  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 --model-path output3/5/best_model.pth --output-dir output3/5  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 --model-path output3/6/best_model.pth --output-dir output3/6  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 --model-path output3/7/best_model.pth --output-dir output3/7  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 --model-path output3/8/best_model.pth --output-dir output3/8  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a --model-path output3/9/best_model.pth --output-dir output3/9  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 --model-path output3/10/best_model.pth --output-dir output3/10  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 --model-path output3/11/best_model.pth --output-dir output3/11  --test-folder test-90% --dg mixstyle # MIXSTYLE
#python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 b12 --model-path output3/12/best_model.pth --output-dir output3/12  --test-folder test-90% --dg mixstyle # MIXSTYLE


python test.py --band-groups b01 b02 --model-path output3/2_pretrained/best_model.pth --output-dir output3/2_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 --model-path output3/3_pretrained/best_model.pth --output-dir output3/3_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 --model-path output3/4_pretrained/best_model.pth --output-dir output3/4_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 --model-path output3/5_pretrained/best_model.pth --output-dir output3/5_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 --model-path output3/6_pretrained/best_model.pth --output-dir output3/6_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 --model-path output3/7_pretrained/best_model.pth --output-dir output3/7_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 --model-path output3/8_pretrained/best_model.pth --output-dir output3/8_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a --model-path output3/9_pretrained/best_model.pth --output-dir output3/9_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 --model-path output3/10_pretrained/best_model.pth --output-dir output3/10_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 --model-path output3/11_pretrained/best_model.pth --output-dir output3/11_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups b01 b02 b03 b04 b05 b06 b07 b08 b8a b09 b11 b12 --model-path output3/12_pretrained/best_model.pth --output-dir output3/12_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE