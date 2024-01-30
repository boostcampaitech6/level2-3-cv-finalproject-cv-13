import React from 'react'
import './TotalResults.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
export default function TotalResults () {
	return (
		<div className='TotalResults_TotalResults'>
			<img className='background' src = {ImgAsset.InputScreen_background} />
			<div className='Overlay'>
				<div className='Buttons'>
					<Link to='/abnormalresultscoronal'>
						<div className='Abnormal'>
							<img className='Rectangle1' src = {ImgAsset.TotalResults_Rectangle1} />
							<span className='Abnormal_1'>Abnormal</span>
						</div>
					</Link>
					<Link to='/aclresultscoronal'>
						<div className='ACL'>
							<img className='Rectangle1_1' src = {ImgAsset.TotalResults_Rectangle1_1} />
							<span className='ACL_1'>ACL</span>
						</div>
					</Link>
					<Link to='/meniscusresultscoronal'>
						<div className='Meniscus'>
							<img className='Rectangle1_2' src = {ImgAsset.TotalResults_Rectangle1_2} />
							<span className='Meniscus_1'>Meniscus</span>
						</div>
					</Link>
					<div className='Total'>
						<img className='Rectangle1_3' src = {ImgAsset.TotalResults_Rectangle1_3} />
						<span className='Total_1'>Total</span>
					</div>
				</div>
				<img className='BubbleA' src = {ImgAsset.TotalResults_BubbleA} />
				<img className='logo' src = {ImgAsset.LoadingScreen_logo} />
				<span className='Text2explain'>The Model thinks that the patient is...</span>
			</div>
			<div className='Contents'>
				<div className='PercentText'>
					<Link to='/abnormalresultscoronal'>
						<span className='AbnormalPercent'>86% Abnormal</span>
					</Link>
					<Link to='/aclresultscoronal'>
						<span className='ACLPercent'>95% ACL</span>
					</Link>
					<Link to='/meniscusresultscoronal'>
						<span className='MeniscusPercent'>27% Meniscus</span>
					</Link>
				</div>
				<div className='Graph'>
					<div className='Graphbox'/>
					<span className='GraphText'>Graph goes here</span>
				</div>
			</div>
		</div>
	)
}