
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
	AxialResults_PlayButton: tryRequire('./AxialResults_PlayButton.png') || require('./questionMark.png'),
	CoronalResults_PlayButton: tryRequire('./CoronalResults_PlayButton.png') || require('./questionMark.png'),
	SagittalResults_PlayButton: tryRequire('./SagittalResults_PlayButton.png') || require('./questionMark.png'),
	AxialResults_PauseButtonWhite: tryRequire('./AxialResults_PauseButtonWhite.png') || require('./questionMark.png'),
	Loading_loading: tryRequire('./Loading_loading.png') || require('./questionMark.png'),
	ResultsCodeACL_logo: tryRequire('./ResultsCodeACL_logo.png') || require('./questionMark.png'),
	FirstImpression_AxialImage: tryRequire('./FirstImpression_AxialImage.png') || require('./questionMark.png'),
	AxialResults_Arrow1: tryRequire('./AxialResults_Arrow1.png') || require('./questionMark.png'),
	CoronalResults_Arrow1: tryRequire('./CoronalResults_Arrow1.png') || require('./questionMark.png'),
	SagittalResults_Arrow1: tryRequire('./SagittalResults_Arrow1.png') || require('./questionMark.png'),
	ResultsCodeACL_Switch: tryRequire('./ResultsCodeACL_Switch.png') || require('./questionMark.png'),
}