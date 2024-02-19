import React from 'react'
import './LoadingScreen.css'
import ImgAsset from '../public'
import { useHistory } from 'react-router-dom';

export default function LoadingScreen () {

	const history = useHistory();

	setTimeout(() => {
		// Code to execute after one second
		console.log("Three seconds have passed");
		history.push("/totalresults")
	  }, 3000); // 1000 milliseconds = 1 second
	

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