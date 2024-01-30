import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import HomePage from '../Components/index';
import InputScreen from '../Components/InputScreen';
import LoadingScreen from '../Components/LoadingScreen';
import AbnormalResultsCoronal from '../Components/AbnormalResultsCoronal';
import AbnormalResultsAxial from '../Components/AbnormalResultsAxial';
import AbnormalResultsSagittal from '../Components/AbnormalResultsSagittal';
import ACLResultsCoronal from '../Components/ACLResultsCoronal';
import MeniscusResultsCoronal from '../Components/MeniscusResultsCoronal';
import MeniscusResultsAxial from '../Components/MeniscusResultsAxial';
import MeniscusResultsSagittal from '../Components/MeniscusResultsSagittal';
import ACLResultsAxial from '../Components/ACLResultsAxial';
import ACLResultsSagittal from '../Components/ACLResultsSagittal';
import TotalResults from '../Components/TotalResults';
const RouterDOM = () => {
	return (
		<Router>
			<Switch>
				<Route exact path="/"><HomePage /></Route>
				<Route exact path="/inputscreen"><InputScreen /></Route>
				<Route exact path="/loadingscreen"><LoadingScreen /></Route>
				<Route exact path="/abnormalresultscoronal"><AbnormalResultsCoronal /></Route>
				<Route exact path="/abnormalresultsaxial"><AbnormalResultsAxial /></Route>
				<Route exact path="/abnormalresultssagittal"><AbnormalResultsSagittal /></Route>
				<Route exact path="/aclresultscoronal"><ACLResultsCoronal /></Route>
				<Route exact path="/meniscusresultscoronal"><MeniscusResultsCoronal /></Route>
				<Route exact path="/meniscusresultsaxial"><MeniscusResultsAxial /></Route>
				<Route exact path="/meniscusresultssagittal"><MeniscusResultsSagittal /></Route>
				<Route exact path="/aclresultsaxial"><ACLResultsAxial /></Route>
				<Route exact path="/aclresultssagittal"><ACLResultsSagittal /></Route>
				<Route exact path="/totalresults"><TotalResults /></Route>
			</Switch>
		</Router>
	);
}
export default RouterDOM;