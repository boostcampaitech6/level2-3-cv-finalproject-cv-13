import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Loading from '../Components/Loading';
import PlaneResultTemplate from '../Components/PlaneResultTemplate';
import TotalResultTemplate from '../Components/TotalResultTemplate';
import InputTemplate from '../Components/InputTemplate';

const RouterDOM = () => {
	return (
		<Router>
			<Switch>
				<Route exact path="/"><InputTemplate inputType='DICOM'/></Route>
				<Route exact path="/PNG"><InputTemplate inputType='Image'/></Route>
				<Route exact path="/results/abnormal"><TotalResultTemplate disease='abnormal' idx='0' /></Route>
				<Route exact path="/results/acl"><TotalResultTemplate disease='acl' idx='1' /></Route>
				<Route exact path="/results/meniscus"><TotalResultTemplate disease='meniscus' idx='2' /></Route>
				<Route exact path="/results/abnormal/axial"><PlaneResultTemplate disease='abnormal' plane='axial' /></Route>
				<Route exact path="/results/abnormal/coronal"><PlaneResultTemplate disease='abnormal' plane='coronal' /></Route>
				<Route exact path="/results/abnormal/sagittal"><PlaneResultTemplate disease='abnormal' plane='sagittal' /></Route>
				<Route exact path="/results/acl/axial"><PlaneResultTemplate disease='acl' plane='axial' /></Route>
				<Route exact path="/results/acl/coronal"><PlaneResultTemplate disease='acl' plane='coronal' /></Route>
				<Route exact path="/results/acl/sagittal"><PlaneResultTemplate disease='acl' plane='sagittal' /></Route>
				<Route exact path="/results/meniscus/axial"><PlaneResultTemplate disease='meniscus' plane='axial' /></Route>
				<Route exact path="/results/meniscus/coronal"><PlaneResultTemplate disease='meniscus' plane='coronal' /></Route>
				<Route exact path="/results/meniscus/sagittal"><PlaneResultTemplate disease='meniscus' plane='sagittal' /></Route>
				<Route exact path="/loading"><Loading /></Route>
			</Switch>
		</Router>
	);
}
export default RouterDOM;