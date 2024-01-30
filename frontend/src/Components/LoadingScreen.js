import React from 'react'
import './LoadingScreen.css'
import ImgAsset from '../public'
export default function LoadingScreen () {
	return (
		<div className='LoadingScreen_LoadingScreen'>
			<img className='background' src = {ImgAsset.InputScreen_background} />
			<img className='logo' src = {ImgAsset.LoadingScreen_logo} />
			<div className='Group1'>
				<div className='imageholder'/>
				<span className='Text'>Loading...</span>
				<img className='loading1' src = {ImgAsset.LoadingScreen_loading1} />
			</div>
		</div>
	)
}