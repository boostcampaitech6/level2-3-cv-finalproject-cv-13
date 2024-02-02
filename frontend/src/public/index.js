
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