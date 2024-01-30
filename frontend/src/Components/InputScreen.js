import React from 'react'
import './InputScreen.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
export default function InputScreen () {
	return (
		<div className='InputScreen_InputScreen'>
			<img className='background' src = {ImgAsset.InputScreen_background} />
			<div className='ImageHolderGroup'>
				<Link to='/loadingscreen'>
					<div className='imageholder'/>
				</Link>
				<img className='Imagelogo' src = {ImgAsset.InputScreen_Imagelogo} />
				<span className='Text'>Place images here<br/>or<br/>Upload from your computer</span>
			</div>
			<img className='logo' src = {ImgAsset.LoadingScreen_logo} />
		</div>
	)
}