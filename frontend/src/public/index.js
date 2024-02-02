
        const tryRequire = (path) => {
        	try {
        	const image = require(`${path}`);
        	return image
        	} catch (err) {
        	return false
        	}
        };

        export default {
        
	questionMark: require('./questionMark.png'),
	InputScreen_background: tryRequire('./InputScreen_background.png') || require('./questionMark.png'),
	LoadingScreen_logo: tryRequire('./LoadingScreen_logo.png') || require('./questionMark.png'),
	TotalResults_Rectangle1: tryRequire('./TotalResults_Rectangle1.png') || require('./questionMark.png'),
	InputScreen_Imagelogo: tryRequire('./InputScreen_Imagelogo.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1: tryRequire('./AbnormalResultsCoronal_Rectangle1.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1: tryRequire('./AbnormalResultsAxial_Rectangle1.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1: tryRequire('./AbnormalResultsSagittal_Rectangle1.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1: tryRequire('./ACLResultsCoronal_Rectangle1.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1: tryRequire('./MeniscusResultsCoronal_Rectangle1.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1: tryRequire('./MeniscusResultsAxial_Rectangle1.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1: tryRequire('./MeniscusResultsSagittal_Rectangle1.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1: tryRequire('./ACLResultsAxial_Rectangle1.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1: tryRequire('./ACLResultsSagittal_Rectangle1.png') || require('./questionMark.png'),
	LoadingScreen_loading1: tryRequire('./LoadingScreen_loading1.png') || require('./questionMark.png'),
	TotalResults_Rectangle1_1: tryRequire('./TotalResults_Rectangle1_1.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_1: tryRequire('./AbnormalResultsCoronal_Rectangle1_1.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_1: tryRequire('./AbnormalResultsAxial_Rectangle1_1.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_1: tryRequire('./AbnormalResultsSagittal_Rectangle1_1.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_1: tryRequire('./ACLResultsCoronal_Rectangle1_1.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_1: tryRequire('./MeniscusResultsCoronal_Rectangle1_1.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_1: tryRequire('./MeniscusResultsAxial_Rectangle1_1.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_1: tryRequire('./MeniscusResultsSagittal_Rectangle1_1.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_1: tryRequire('./ACLResultsAxial_Rectangle1_1.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_1: tryRequire('./ACLResultsSagittal_Rectangle1_1.png') || require('./questionMark.png'),
	TotalResults_Rectangle1_2: tryRequire('./TotalResults_Rectangle1_2.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_2: tryRequire('./AbnormalResultsCoronal_Rectangle1_2.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_2: tryRequire('./AbnormalResultsAxial_Rectangle1_2.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_2: tryRequire('./AbnormalResultsSagittal_Rectangle1_2.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_2: tryRequire('./ACLResultsCoronal_Rectangle1_2.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_2: tryRequire('./MeniscusResultsCoronal_Rectangle1_2.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_2: tryRequire('./MeniscusResultsAxial_Rectangle1_2.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_2: tryRequire('./MeniscusResultsSagittal_Rectangle1_2.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_2: tryRequire('./ACLResultsAxial_Rectangle1_2.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_2: tryRequire('./ACLResultsSagittal_Rectangle1_2.png') || require('./questionMark.png'),
	TotalResults_Rectangle1_3: tryRequire('./TotalResults_Rectangle1_3.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_3: tryRequire('./AbnormalResultsCoronal_Rectangle1_3.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_3: tryRequire('./AbnormalResultsAxial_Rectangle1_3.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_3: tryRequire('./AbnormalResultsSagittal_Rectangle1_3.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_3: tryRequire('./ACLResultsCoronal_Rectangle1_3.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_3: tryRequire('./MeniscusResultsCoronal_Rectangle1_3.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_3: tryRequire('./MeniscusResultsAxial_Rectangle1_3.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_3: tryRequire('./MeniscusResultsSagittal_Rectangle1_3.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_3: tryRequire('./ACLResultsAxial_Rectangle1_3.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_3: tryRequire('./ACLResultsSagittal_Rectangle1_3.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_4: tryRequire('./AbnormalResultsCoronal_Rectangle1_4.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_4: tryRequire('./AbnormalResultsAxial_Rectangle1_4.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_4: tryRequire('./AbnormalResultsSagittal_Rectangle1_4.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_4: tryRequire('./ACLResultsCoronal_Rectangle1_4.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_4: tryRequire('./MeniscusResultsCoronal_Rectangle1_4.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_4: tryRequire('./MeniscusResultsAxial_Rectangle1_4.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_4: tryRequire('./MeniscusResultsSagittal_Rectangle1_4.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_4: tryRequire('./ACLResultsAxial_Rectangle1_4.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_4: tryRequire('./ACLResultsSagittal_Rectangle1_4.png') || require('./questionMark.png'),
	TotalResults_BubbleA: tryRequire('./TotalResults_BubbleA.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Overlay: tryRequire('./AbnormalResultsCoronal_Overlay.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Overlay: tryRequire('./AbnormalResultsAxial_Overlay.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Overlay: tryRequire('./AbnormalResultsSagittal_Overlay.png') || require('./questionMark.png'),
	ACLResultsCoronal_Overlay: tryRequire('./ACLResultsCoronal_Overlay.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Overlay: tryRequire('./MeniscusResultsCoronal_Overlay.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Overlay: tryRequire('./MeniscusResultsAxial_Overlay.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Overlay: tryRequire('./MeniscusResultsSagittal_Overlay.png') || require('./questionMark.png'),
	ACLResultsAxial_Overlay: tryRequire('./ACLResultsAxial_Overlay.png') || require('./questionMark.png'),
	ACLResultsSagittal_Overlay: tryRequire('./ACLResultsSagittal_Overlay.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_PlayButton: tryRequire('./AbnormalResultsCoronal_PlayButton.png') || require('./questionMark.png'),
	AbnormalResultsAxial_PlayButton: tryRequire('./AbnormalResultsAxial_PlayButton.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_PlayButton: tryRequire('./AbnormalResultsSagittal_PlayButton.png') || require('./questionMark.png'),
	ACLResultsCoronal_PlayButton: tryRequire('./ACLResultsCoronal_PlayButton.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_PlayButton: tryRequire('./MeniscusResultsCoronal_PlayButton.png') || require('./questionMark.png'),
	MeniscusResultsAxial_PlayButton: tryRequire('./MeniscusResultsAxial_PlayButton.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_PlayButton: tryRequire('./MeniscusResultsSagittal_PlayButton.png') || require('./questionMark.png'),
	ACLResultsAxial_PlayButton: tryRequire('./ACLResultsAxial_PlayButton.png') || require('./questionMark.png'),
	ACLResultsSagittal_PlayButton: tryRequire('./ACLResultsSagittal_PlayButton.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_GradCAMButton: tryRequire('./AbnormalResultsCoronal_GradCAMButton.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_5: tryRequire('./AbnormalResultsAxial_Rectangle1_5.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_5: tryRequire('./AbnormalResultsSagittal_Rectangle1_5.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_5: tryRequire('./ACLResultsCoronal_Rectangle1_5.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_5: tryRequire('./MeniscusResultsCoronal_Rectangle1_5.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_5: tryRequire('./MeniscusResultsAxial_Rectangle1_5.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_5: tryRequire('./MeniscusResultsSagittal_Rectangle1_5.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_5: tryRequire('./ACLResultsAxial_Rectangle1_5.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_5: tryRequire('./ACLResultsSagittal_Rectangle1_5.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_5: tryRequire('./AbnormalResultsCoronal_Rectangle1_5.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_6: tryRequire('./AbnormalResultsAxial_Rectangle1_6.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_6: tryRequire('./AbnormalResultsSagittal_Rectangle1_6.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_6: tryRequire('./ACLResultsCoronal_Rectangle1_6.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_6: tryRequire('./MeniscusResultsCoronal_Rectangle1_6.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_6: tryRequire('./MeniscusResultsAxial_Rectangle1_6.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_6: tryRequire('./MeniscusResultsSagittal_Rectangle1_6.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_6: tryRequire('./ACLResultsAxial_Rectangle1_6.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_6: tryRequire('./ACLResultsSagittal_Rectangle1_6.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_6: tryRequire('./AbnormalResultsCoronal_Rectangle1_6.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_7: tryRequire('./AbnormalResultsAxial_Rectangle1_7.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_7: tryRequire('./AbnormalResultsSagittal_Rectangle1_7.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_7: tryRequire('./ACLResultsCoronal_Rectangle1_7.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_7: tryRequire('./MeniscusResultsCoronal_Rectangle1_7.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_7: tryRequire('./MeniscusResultsAxial_Rectangle1_7.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_7: tryRequire('./MeniscusResultsSagittal_Rectangle1_7.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_7: tryRequire('./ACLResultsAxial_Rectangle1_7.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_7: tryRequire('./ACLResultsSagittal_Rectangle1_7.png') || require('./questionMark.png'),
	AbnormalResultsCoronal_Rectangle1_7: tryRequire('./AbnormalResultsCoronal_Rectangle1_7.png') || require('./questionMark.png'),
	AbnormalResultsAxial_Rectangle1_8: tryRequire('./AbnormalResultsAxial_Rectangle1_8.png') || require('./questionMark.png'),
	AbnormalResultsSagittal_Rectangle1_8: tryRequire('./AbnormalResultsSagittal_Rectangle1_8.png') || require('./questionMark.png'),
	ACLResultsCoronal_Rectangle1_8: tryRequire('./ACLResultsCoronal_Rectangle1_8.png') || require('./questionMark.png'),
	MeniscusResultsCoronal_Rectangle1_8: tryRequire('./MeniscusResultsCoronal_Rectangle1_8.png') || require('./questionMark.png'),
	MeniscusResultsAxial_Rectangle1_8: tryRequire('./MeniscusResultsAxial_Rectangle1_8.png') || require('./questionMark.png'),
	MeniscusResultsSagittal_Rectangle1_8: tryRequire('./MeniscusResultsSagittal_Rectangle1_8.png') || require('./questionMark.png'),
	ACLResultsAxial_Rectangle1_8: tryRequire('./ACLResultsAxial_Rectangle1_8.png') || require('./questionMark.png'),
	ACLResultsSagittal_Rectangle1_8: tryRequire('./ACLResultsSagittal_Rectangle1_8.png') || require('./questionMark.png'),

	sample0: tryRequire('./axial_0000.png') || require('./questionMark.png'),
	sample1: tryRequire('./axial_0001.png') || require('./questionMark.png'),
	sample2: tryRequire('./axial_0002.png') || require('./questionMark.png'),
	sample3: tryRequire('./axial_0003.png') || require('./questionMark.png'),
	sample4: tryRequire('./axial_0004.png') || require('./questionMark.png'),
	sample5: tryRequire('./axial_0005.png') || require('./questionMark.png'),
	sample6: tryRequire('./axial_0006.png') || require('./questionMark.png'),
	sample7: tryRequire('./axial_0007.png') || require('./questionMark.png'),
	sample8: tryRequire('./axial_0008.png') || require('./questionMark.png'),
	sample9: tryRequire('./axial_0009.png') || require('./questionMark.png'),
}