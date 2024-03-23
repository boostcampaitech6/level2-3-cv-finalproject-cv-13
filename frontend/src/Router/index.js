import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import {useState, useEffect} from 'react';
import { HmacSHA256 } from 'crypto-js';
import Loading from '../Components/Loading';
import PlaneResultTemplate from '../Components/PlaneResultTemplate';
import TotalResultTemplate from '../Components/TotalResultTemplate';
import InputTemplate from '../Components/InputTemplate';

const RouterDOM = () => {

	const [ip, setIP] = useState("");

	const getIP = () => {
	    // use JSONP response to bypass CORS error
	    // create <script> src = url <script/>
		const script = document.createElement('script');
		script.src = `https://geolocation-db.com/jsonp/`;
		
		// callback looks like this
		/* 
			callback({
			"country_code":"US",
			"country_name":"United States",
			"city":"Minneapolis",
			"postal":55455,
			"latitude":44.9733,
			"longitude":-93.2323,
			"IPv4":"126.101.76.251",
			"state":"Minnesota"
			}) 
		*/
		window.callback = (data) => {
			const real_ip = data.IPv4;
			const key = 'lukeh';
			const hash_ip = HmacSHA256(real_ip, key).toString();
			setIP(hash_ip);
		};
  
	    document.body.appendChild(script);
	};
  
	useEffect(() => {
	    getIP();
		localStorage.setItem('userIP', ip);
	}, [ip]);

	return (
		<Router>
			<Switch>
				<Route exact path="/"><InputTemplate inputType='DICOM' ip={ip} /></Route>
				<Route exact path="/results/abnormal"><TotalResultTemplate disease='abnormal' idx='0' ip={ip} /></Route>
				<Route exact path="/results/acl"><TotalResultTemplate disease='acl' idx='1' ip={ip}/></Route>
				<Route exact path="/results/meniscus"><TotalResultTemplate disease='meniscus' idx='2' ip={ip}/></Route>
				<Route exact path="/results/abnormal/axial"><PlaneResultTemplate disease='abnormal' plane='axial' ip={ip}/></Route>
				<Route exact path="/results/abnormal/coronal"><PlaneResultTemplate disease='abnormal' plane='coronal' ip={ip}/></Route>
				<Route exact path="/results/abnormal/sagittal"><PlaneResultTemplate disease='abnormal' plane='sagittal' ip={ip}/></Route>
				<Route exact path="/results/acl/axial"><PlaneResultTemplate disease='acl' plane='axial' ip={ip}/></Route>
				<Route exact path="/results/acl/coronal"><PlaneResultTemplate disease='acl' plane='coronal' ip={ip}/></Route>
				<Route exact path="/results/acl/sagittal"><PlaneResultTemplate disease='acl' plane='sagittal' ip={ip}/></Route>
				<Route exact path="/results/meniscus/axial"><PlaneResultTemplate disease='meniscus' plane='axial' ip={ip}/></Route>
				<Route exact path="/results/meniscus/coronal"><PlaneResultTemplate disease='meniscus' plane='coronal' ip={ip}/></Route>
				<Route exact path="/results/meniscus/sagittal"><PlaneResultTemplate disease='meniscus' plane='sagittal' ip={ip}/></Route>
				<Route exact path="/loading"><Loading ip={ip}/></Route>
			</Switch>
		</Router>
	);
}
export default RouterDOM;