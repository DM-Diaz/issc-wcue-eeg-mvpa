This contains code AND data used for ERP/MVPA analyses for the ISSC_wCue experiment 2026 during the Purdue SROP program and presented at the 2026 Purdue Summer Research Conference.

FULL DOCUMENTATION: ISSC_wCue_EEG_Methods_and_Analysis_DM_Diaz_YC_Chiu_2026.pdf for (exhaustive) details.
GITHUB REPO: https://github.com/DM-Diaz/issc-wcue-eeg-mvpa

Researcher: Dylan M. Diaz (Institutional email: dylan.diaz4811@coyote.csusb.edu; ORCID: https://orcid.org/0009-0002-0872-1200)
PI: Yu-Chin Chiu (chiu56@purdue.edu)

**Original Abstract (see related poster_Dylan_M_Diaz_Yu_Chin_Chiu_2026_SROP.pdf):
Cognitive control enables us to flexibly switch between tasks, although doing so is effortful and costly: laboratory studies consistently show that switch trials produce slower responses and more errors than repeat trials. When given the freedom to choose, people also tend to repeat the task they have just performed rather than switch to a different one. Braem (2017) recently found that participants rewarded for task switches showed greater subsequent voluntary switching than participants rewarded for task repetitions. One interpretation is that participants learned an association between reward and switching, making them more willing to switch later. However, because that study compared separate groups of participants, it remains unclear whether participants learned a general preference for switching or a specific association between reward and a particular control operation. Here, we tested whether rewards can shape specific cognitive control operations—switching versus repeating—by manipulating reward contingencies within individuals. In Experiment 1, using behavioral measures, we found that participants learned the specific association between individual items and the rewarded control operation (switch or repeat). Surprisingly, however, this learning produced larger switch costs for switch-rewarded items than for repeat-rewarded items. In Experiment 2, we extend this work to the neural level using EEG recordings and cross-validated multivariate pattern analysis which reads patterns of neural activity. This will allow us to test whether distinct brain states track these reward conditions and predict the slowdown we see in behavior. Together, this work refines how we understand reward's role in balancing mental stability against flexibility, and shows that this tuning can occur locally, item by item, and within individuals.

==========================================================================================================

Programs used:
- Matlab R2024a # this was the IDE used to run Matlab code for EEG data
- PyCharm 2026.1  # this was the IDE used to run Python code for MVPA
- Python 3.12.3

For full list of packages used for MVPA analyses in Python see MVPA_requirements.txt in MVPA_Python folder
For full list of toolboxes used for EEG processing in Matlab see Matlab_requirements.txt in EEG_Matlab

==========================================================================================================
Experimental structure:

block #		trialID		bkSRProb	item
1		1		High		3,2
2		97		Low		1,2
3		193		Neutral		1,3
4		289		Low		1,2
5		385		High		3,2
--------------------------------------------------------------
6		481				Hybrid (1&3)
7		577				Hybrid (1&3)

(Blocks 1-5 were used for this analysis/project)

"item"
1 rpCond
2 Diagnostic
3 swCond

"task cue"
4, 5 (switch)
6, 7 (repeat)

See ISSC_trigger_codes.txt and binlister files (in EEG_Matlab folder) for more relevant structure info.

Lastly, to reiterate: see the full documentation "ISSC_wCue_EEG_Methods_and_Analysis_DM_Diaz_YC_Chiu_2026.pdf" for (exhaustive) details.

