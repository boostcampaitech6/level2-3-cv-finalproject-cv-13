import React from 'react'
import './SagittalResults.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
export default function SagittalResults () {
	return (
		<div className="sagittal-results">
      <div className="div">
        <img className="play-button" alt="Play button" src="play-button.svg" />
        <img className="pause-button-white" alt="Pause button white" src="pause-button-white.png" />
        <div className="graph">
          <div className="overlap-group">
            <div className="text-wrapper">Graph goes here</div>
          </div>
        </div>
        <div className="image">
          <div className="overlap">
            <div className="image-text">Image goes here</div>
          </div>
        </div>
        <div className="top-overlay">
          <div className="overlap-2">
            <div className="overlap-group-2">
              <div className="text-wrapper-2">환자명: 김00</div>
              <div className="text-wrapper-3">성별: F</div>
              <div className="text-wrapper-4">검사일: 2024-02-27</div>
            </div>
            <img className="logo" alt="Logo" src="logo.png" />
          </div>
        </div>
        <div className="slider">
          <div className="div-wrapper">
            <div className="text-wrapper-5">Slider goes here</div>
          </div>
        </div>
        <div className="back">
          <div className="overlap-3">
            <div className="ellipse" />
            <img className="arrow" alt="Arrow" src="arrow-1.svg" />
          </div>
        </div>
      </div>
    </div>
	)
}