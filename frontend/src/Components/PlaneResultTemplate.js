import React, {useRef, useState, useEffect} from 'react'
import axios from 'axios'
import './PlaneResultTemplate.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import {Button} from "@mui/material";
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend,
  } from 'chart.js';
import { Line, getElementAtEvent } from 'react-chartjs-2';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend,
	annotationPlugin
);

export default function PlaneResultTemplate (props) {

  	let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState([]);
	const [loading, setLoading] = useState(true);
	const [onPatientLoad, setonPatientLoad] = useState(true);
	const chartRef = useRef();
	const [Data, setData] = useState([]);
	let [pauseState, setPauseState] = useState(false);
	const [patientInfo, setPatientInfo] = useState([]);
	const [patientLabel, setPatientLabel] = useState([]);

	const disease = props.disease;
	const plane = props.plane;
	const ip = localStorage.getItem('userIP') || props.ip;
	const imageurl = `/output/${disease}/${plane}`;
	const patienturl = "/result/patient";
	const config = {
		headers: {
		  'IP': ip,
		}
	  };

	useEffect(() => {
		async function fetchData() {
		  try {
			const imageResponse = await axios.get(imageurl, config);
			setImages(imageResponse.data.img);
			setData(imageResponse.data.info);
			setImageExists(true);
			setLoading(false);
		  } catch (error) {
			alert(`이미지 로딩시 에러가 발생했습니다. \n ${error}`);
			setImageExists(false);
			setLoading(false);
		  }
		}
		fetchData();
	  }, []);

	useEffect(() => {
	const fetchPatient = async () => {
			try {
			const response = await axios.get(patienturl, config)
			setPatientInfo(response.data.info);
			setPatientLabel(response.data.labels);
			setonPatientLoad(false);
			} catch (error) {
			alert(`환자정보 로딩 시 에러가 발생했습니다. \n ${error}`);
			setonPatientLoad(true);
			}
		};
		fetchPatient();
		}, []);
	

	let [page, setPage] = useState(images);
	let [currentidx, setCurrentIdx] = useState(loading ? 0 : Data.highest);
	
	useEffect(() => {
	setPage(images);
	}, [images]);

	useEffect(() => {
	// Ensure currentidx is within bounds
	if (currentidx < 0) setCurrentIdx(0);
	if (currentidx >= page.length && page.length != 0) setCurrentIdx(page.length - 1);
	if (page.length == 0) setCurrentIdx(0);
	}, [currentidx, page]);

	const handleImageError = () => {
		setImageExists(false); // Set imageExists to false if the image fails to load
	  };
	
	let labels = [];
    if (Data.labels) {
        labels = Data.labels;
    }

	const dataset = {
		labels,
		datasets: [
			{
				label: "Importance of Slides",
				data: Data.datasets,
				borderColor: "rgb(255, 255, 255",
				backgroundColor: "rgba(255, 255, 255, 0.8)",
			},
		],
	};

	function handleChartClick (event) {
		const element = getElementAtEvent(chartRef.current, event);
		const index = element[0].index;
		setCurrentIdx(index);
	  };

	useEffect(() => {
		let intervalId;
	
		if (pauseState) {
		  intervalId = setInterval(() => {
			if (currentidx < Data.datasets.length - 1) {
			  setCurrentIdx(prevIdx => prevIdx + 1);
			} else {
			  clearInterval(intervalId);
			}
		  }, 100);
		}
	
		return () => clearInterval(intervalId);
	  }, [currentidx, pauseState]);
	
	function playButton (e) {
		e.preventDefault();
		setPauseState(true);
	}

	function pauseButton (e) {
		e.preventDefault();
		setPauseState(false);
	}

	const options = {
		aspectRatio: 6.3,
		responsive: true,
		plugins: {
		  legend: {
			position: 'top',
		  },
		  title: {
			display: true,
			text: 'Axial Scores',
		  },
		},
		plugins: {
			annotation: {
			  annotations: {
				line1: {
				  type: 'line',
				  xMin: currentidx,
				  xMax: currentidx,
				  borderColor:'rgb(255, 255, 255)',
				  borderWidth: 2,
				}
			  }
			}
		  }
	  };

	return (
	<div className="axial-results">
      <div className="div">
        <div className="graph">
            <Line options={options} data={dataset} ref={chartRef} onClick={handleChartClick} />
        </div>
        <div className="image">
		<div className="overlap">
              {loading ? (
              <span>Loading...</span>
            ) : imageExists ? (
              page[currentidx] ? (
                <img
                  className="overlapimage"
                  alt="Press Inspect to Start!"
                  key={currentidx}
                  src={`data:image/png;base64,${page[currentidx].body}`}
                  onError={handleImageError}
                />
              ) : (
                <span>Image not found</span>
              )
            ) : (
              <span>Error loading images</span>
            )}
        </div>
        </div>
        <div className="top-overlay">
          <div className="overlap-2">
            <div className="overlap-group-2">
              <div className="text-wrapper-2">{!onPatientLoad ? `${patientLabel[0]}: ${patientInfo[0]} \u00A0 ${patientLabel[1]}: ${patientInfo[1]} \u00A0 
              ${patientLabel[2]}: ${patientInfo[2]} \u00A0 ${patientLabel[3]}: ${patientInfo[3]} \u00A0 ${patientLabel[4]}: ${patientInfo[4]}` : 'Loading...'}</div>
            </div>
			<Link to="/">
            <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo}/>
			</Link>
          </div>
        </div>
		<div className="play-button">
			<Button  variant="text" onClick={playButton} sx = {{backgroundImage:`url(${ImgAsset.AxialResults_PlayButton})`, backgroundRepeat: "no-repeat", 
			width: "35px", height: "49px",backgroundSize: '35px', backgroundPosition: 'center',}}></Button>
			</div>
		<div className="pause-button-white">
			<Button  variant="text" onClick={pauseButton} sx = {{backgroundImage:`url(${ImgAsset.AxialResults_PauseButtonWhite})`, backgroundRepeat: "no-repeat", 
			width: "35px", height: "49px",backgroundSize: '28px', backgroundPosition: 'center',}}></Button>
		</div>
        <div className="slider">
              <button style={{backgroundColor: 'black', color: 'white'}}
              className="nav-btn1"
              onClick={() => {
              if (currentidx === 0) {
                currentidx = currentidx + 1;
              }
              setCurrentIdx(currentidx - 1);
              }}
            >
              <NavigateBeforeIcon />
            </button>
            <button style={{backgroundColor: 'black', color: 'white'}}
              className="nav-btn2"
              onClick={() => {
              if (currentidx === Data.datasets.length - 1) {
                currentidx = currentidx - 1;
              }
              setCurrentIdx(currentidx + 1);
              }}
            >
              <NavigateNextIcon />
            </button>
        </div>
		<Link to={`/results/${disease}`}>
        <div className="back">
            <div className="ellipse" />
            <img className="arrow" alt="Arrow" src={ImgAsset.AxialResults_Arrow1} />
        </div>
		</Link>
      </div>
    </div>
	)
}