import React from 'react'
import './AbnormalResultsCoronal.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
export default function AbnormalResultsCoronal () {
	return (
		<div className='AbnormalResultsCoronal_AbnormalResultsCoronal'>
			<img className='background' src = {ImgAsset.background} />
			<img className='logo' src = {ImgAsset.logo} />
			{/* <div className='Coronal'>
				<img className='Rectangle1' src = {ImgAsset.AbnormalResultsCoronal_Rectangle1} />
				<span className='GradCAM'>Grad-CAM</span>
			</div> */}
			<div className='OuterButtons'>
				<div className='Axial'>
					<img className='Rectangle1_1' src = {ImgAsset.blackrectangle} />
					<span className='Abnormal'>Abnormal</span>
				</div>
				<Link to='/aclresultscoronal'>
					<div className='ACLButton'>
						<img className='Rectangle1_2' src = {ImgAsset.whiterectangle} />
						<span className='ACL'>ACL</span>
					</div>
				</Link>
				<Link to='/meniscusresultscoronal'>
					<div className='MeniscusButton'>
						<img className='Rectangle1_3' src = {ImgAsset.whiterectangle} />
						<span className='Meniscus'>Meniscus</span>
					</div>
				</Link>
				<Link to='/totalresults'>
					<div className='TotalButton'>
						<img className='Rectangle1_4' src = {ImgAsset.whiterectangle} />
						<span className='Total'>Total</span>
					</div>
				</Link>
			</div>
			<img className='Overlay' src = {ImgAsset.overlay1} />
			<img className='PlayButton' src = {ImgAsset.playbutton} />
			<div className='PauseButton'>
				<div className='Rectangle6'/>
				<div className='Rectangle7'/>
			</div>
			<div className='Buttons'>
				<img className='GradCAMButton' src = {ImgAsset.blackrectangle} />
				<span className='GradCAM_1'>Grad-CAM</span>
				<div className='InnerButtons'>
					<div className='CoronalButton'>
						<img className='Rectangle1_5' src = {ImgAsset.blackrectangle} />
						<span className='Coronal_1'>Coronal</span>
					</div>
					<Link to='/abnormalresultsaxial'>
						<div className='AxialButton'>
							<img className='Rectangle1_6' src = {ImgAsset.whiterectangle} />
							<span className='Axial_1'>Axial</span>
						</div>
					</Link>
					<Link to='/abnormalresultssagittal'>
						<div className='SagittalButton'>
							<img className='Rectangle1_7' src = {ImgAsset.whiterectangle} />
							<span className='Sagittal'>Sagittal</span>
						</div>
					</Link>
				</div>
			</div>
			<div className='Slider'>
				<div className='SliderBox'/>
				<span className='Slidergoeshere'>Slider goes here</span>
			</div>
			<div className='Graph'>
				<div className='GraphBox'/>
				<span className='Graphgoeshere'>Graph goes here</span>
			</div>
			<div className='Image'>
				<div className='Imagebox'/>
				<span className='ImageText'>Image goes here</span>
			</div>
		</div>
	)
}