import React from 'react'
import './LoadingScreen.css'
import ImgAsset from '../public'
export default function LoadingScreen () {
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