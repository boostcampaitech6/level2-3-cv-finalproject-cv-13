
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
	background: tryRequire('./Background.png') || require('./questionMark.png'),
	blackrectangle: tryRequire('./BlackRectangle.png') || require('./questionMark.png'),
	whiterectangle: tryRequire('./WhiteRectangle.png') || require('./questionMark.png'),
	imageicon: tryRequire('./Imagelogo.png') || require('./questionMark.png'),
	loadingicon: tryRequire('./Loading.png') || require('./questionMark.png'),
	logo: tryRequire('./Logo.png') || require('./questionMark.png'),
	overlay1: tryRequire('./Overlay1.png') || require('./questionMark.png'),
	overlay2: tryRequire('./Overlay2.png') || require('./questionMark.png'),
	overlay3: tryRequire('./Overlay3.png') || require('./questionMark.png'),
	overlay4: tryRequire('./Overlay4.png') || require('./questionMark.png'),
	playbutton: tryRequire('./Playbutton.png') || require('./questionMark.png'),
	pausebutton: tryRequire('./PauseButton.png') || require('./questionMark.png'),
}