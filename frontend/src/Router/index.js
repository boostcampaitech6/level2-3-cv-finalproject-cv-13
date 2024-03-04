import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
// import HomePage from '../Components/index';
import ResultsCodeACL from '../Components/ResultsCodeACL';
import ResultsCodeAbnormal from '../Components/ResultsCodeAbnormal';
import ResultsCodeMeniscus from '../Components/ResultsCodeMeniscus';
import AxialResults from '../Components/AxialResults';
import CoronalResults from '../Components/CoronalResults';
import SagittalResults from '../Components/SagittalResults';
import FirstImpression from '../Components/FirstImpression';
import Loading from '../Components/Loading';
const RouterDOM = () => {
	return (
		<Router>
			<Switch>
				<Route exact path="/"><FirstImpression /></Route>
				<Route exact path="/resultscodeacl"><ResultsCodeACL /></Route>
				<Route exact path="/resultscodeabnormal"><ResultsCodeAbnormal /></Route>
				<Route exact path="/resultscodemeniscus"><ResultsCodeMeniscus /></Route>
				<Route exact path="/axialresults"><AxialResults /></Route>
				<Route exact path="/coronalresults"><CoronalResults /></Route>
				<Route exact path="/sagittalresults"><SagittalResults /></Route>
				{/* <Route exact path="/firstimpression"><FirstImpression /></Route> */}
				<Route exact path="/loading"><Loading /></Route>
			</Switch>
		</Router>
	);
}
export default RouterDOM;