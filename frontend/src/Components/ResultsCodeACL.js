import React from 'react'
import './ResultsCodeACL.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
export default function ResultsCodeACL () {
	return (
		<div className="results-code-ACL">
      <div className="div">
        <div className="top-overlay">
          <div className="overlap-group">
            <div className="overlap">
              <div className="text-wrapper">환자명: 김00</div>
              <div className="text-wrapper-2">성별: F</div>
              <div className="text-wrapper-3">검사일: 2024-02-27</div>
            </div>
            <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo} />
          </div>
        </div>
        <div className="overlap-2">
          <div className="abnormal-score">
            <div className="overlap-3">
              <div className="text-wrapper-4">Abnormality Score</div>
              <div className="rectangle" />
              <div className="text-wrapper-5">86</div>
              <div className="text-wrapper-6">%</div>
            </div>
          </div>
          <div className="category">
            <div className="text-wrapper-7">Category</div>
            <img className="switch" alt="Switch" src={ImgAsset.ResultsCodeACL_Switch} />
            <div className="text-wrapper-8">Show</div>
            <div className="abnormal-button">
              <div className="overlap-group-2">
                <div className="abnormal-cat" />
                <div className="text-wrapper-9">Abnormal</div>
              </div>
            </div>
            <div className="meniscus-button">
              <div className="overlap-group-2">
                <div className="meniscus-cat" />
                <div className="text-wrapper-9">Meniscus</div>
              </div>
            </div>
            <div className="ACL-button">
              <div className="div-wrapper">
                <div className="text-wrapper-10">ACL</div>
              </div>
            </div>
          </div>
        </div>
        <div className="ACL-score">
          <div className="overlap-4">
            <div className="text-wrapper-11">ACL Score</div>
            <div className="overlap-5">
              <div className="rectangle-2" />
              <div className="text-wrapper-12">86</div>
              <div className="text-wrapper-13">%</div>
            </div>
          </div>
        </div>
        <div className="meniscus-score">
          <div className="overlap-6">
            <div className="text-wrapper-14">Meniscus Score</div>
            <div className="overlap-7">
              <div className="rectangle-2" />
              <div className="text-wrapper-12">86</div>
              <div className="text-wrapper-13">%</div>
            </div>
          </div>
        </div>
        <div className="result">
          <div className="text-wrapper-15">Result</div>
          <div className="overlap-8">
            <div className="text-wrapper-16">Normal</div>
          </div>
          <div className="overlap-9">
            <div className="text-wrapper-16">Others</div>
          </div>
          <div className="overlap-10">
            <div className="text-wrapper-17">Meniscus</div>
          </div>
          <div className="overlap-11">
            <div className="text-wrapper-18">ACL</div>
          </div>
        </div>
        <div className="axial-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              <div className="overlap-group-3">
                <div className="graph-text">Axial</div>
                <div className="heatmap-bar">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-19">0</div>
                  <div className="text-wrapper-20">100</div>
                </div>
              </div>
            </div>
            <div className="inspect-ax">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            <div className="plus-ax">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
        <div className="coronal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              <div className="overlap-group-3">
                <div className="graph-text">Coronal</div>
                <div className="heatmap-bar-2">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-19">0</div>
                  <div className="text-wrapper-20">100</div>
                </div>
              </div>
            </div>
            <div className="inspect-co">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            <div className="plus-co">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
        <div className="sagittal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              <div className="overlap-group-3">
                <div className="graph-text">Sagittal</div>
                <div className="heatmap-bar-3">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-19">0</div>
                  <div className="text-wrapper-20">100</div>
                </div>
              </div>
            </div>
            <div className="inspect-sag">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            <div className="plus-sag">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
	)
}