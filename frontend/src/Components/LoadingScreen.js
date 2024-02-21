import React, {useEffect} from 'react'
import axios from 'axios'
import './LoadingScreen.css'
import ImgAsset from '../public'
import { useHistory } from 'react-router-dom';

export default function LoadingScreen () {

	const history = useHistory();

	useEffect(() => {
		const awaitInference = async () => {
			try {
				const response = await axios.get("http://127.0.0.1:8000/inference");
				alert("Inference Done");
				history.push("/totalresults")
				} catch (error) {
					console.log(error);
					alert("Error occured during inference");
				}
		};
		awaitInference();
	}, []);

	return (
		<div className='LoadingScreen_LoadingScreen'>
			<img className='background' src = {ImgAsset.background} />
			<img className='logo' src = {ImgAsset.logo} />
			<div className='Group1'>
				<div className='imageholder'/>
				<span className='Text'>Loading...</span>
				<img className='loading1' src = {ImgAsset.loadingicon} />
			</div>
		</div>
	)
}