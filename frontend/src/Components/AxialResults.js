import React, {useRef, useState, useEffect} from 'react'
import axios from 'axios'
import './AxialResults.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
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

export default function AxialResults () {

  	let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState([]);
	const [loading, setLoading] = useState(true);
	const chartRef = useRef();
	const [Data, setData] = useState([]);
	let [pauseState, setPauseState] = useState(false);

	useEffect(() => {
		async function fetchData() {
		  try {
			// const [imageResponse, gradImageResponse] = await Promise.all([
			// 	axios.get('http://127.0.0.1:8000/output/abnormal/axial/original'),
			//   ]);
			const imageResponse = await axios.get('http://127.0.0.1:8000/output/abnormal/axial/original')
			setImages(imageResponse.data.img);
			setData(imageResponse.data.info);
			setImageExists(true);
			setLoading(false);
		  } catch (error) {
			console.error('Error fetching images:', error);
			setImageExists(false);
			setLoading(false);
		  }
		}
	
		fetchData();
	  }, []);

	let [page, setPage] = useState(images);
	let [currentidx, setCurrentIdx] = useState(loading ? 0 : Data.highest);
	
	useEffect(() => {
	// Set the initial page based on gradstate
	setPage(images);
	}, [images]);

	useEffect(() => {
	// Ensure currentidx is within bounds
	if (currentidx < 0) setCurrentIdx(0);
	if (currentidx >= page.length) setCurrentIdx(page.length - 1);
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
				borderColor: "rgb(255, 99, 132)",
				backgroundColor: "rgba(255, 99, 132, 0.5)",
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
		  }, 300);
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
				  borderColor: 'rgb(255, 99, 132)',
				  borderWidth: 2,
				}
			  }
			}
		  }
	  };

	return (
	<div className="axial-results">
      <div className="div">
        <div className="play-button">
          <Button  variant="text" onClick={playButton} sx = {{backgroundImage:`url(${ImgAsset.AxialResults_PlayButton})`, backgroundRepeat: "no-repeat", 
          width: "35px", height: "49px",backgroundSize: '35px', backgroundPosition: 'center',}}></Button>
        </div>
        <div className="pause-button-white">
          <Button  variant="text" onClick={pauseButton} sx = {{backgroundImage:`url(${ImgAsset.AxialResults_PauseButtonWhite})`, backgroundRepeat: "no-repeat", 
          width: "35px", height: "49px",backgroundSize: '28px', backgroundPosition: 'center',}}></Button>
        </div>
        <div className="graph">
          {/* <div className="overlap-group"> */}
            <Line options={options} data={dataset} ref={chartRef} onClick={handleChartClick} />
          {/* </div> */}
        </div>
        <div className="image">
          {/* <div className="overlap"> */}
              {loading ? (
              <span>Loading...</span>
            ) : imageExists ? (
              page[currentidx] ? (
                <img
                  className="overlap"
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
          {/* </div> */}
        </div>
        <div className="top-overlay">
          <div className="overlap-2">
            <div className="overlap-group-2">
              <div className="text-wrapper-2">환자명: 김00</div>
              <div className="text-wrapper-3">성별: F</div>
              <div className="text-wrapper-4">검사일: 2024-02-27</div>
            </div>
			<Link to="/">
            <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo}/>
			</Link>
          </div>
        </div>
        <div className="slider">
          {/* <div className="div-wrapper"> */}
              <button
              className="nav-btn1"
              onClick={() => {
              if (currentidx === 0) {
                currentidx = currentidx + 1;
              }
              setCurrentIdx(currentidx - 1);
              }}
            >
              <ChevronLeftIcon />
            </button>
            <button
              className="nav-btn2"
              onClick={() => {
              if (currentidx === Data.datasets.length - 1) {
                currentidx = currentidx - 1;
              }
              setCurrentIdx(currentidx + 1);
              }}
            >
              <ChevronRightIcon />
            </button>
          {/* </div> */}
        </div>
		<Link to="/resultscodeabnormal">
        <div className="back">
          <div className="overlap-3">
            <div className="ellipse" />
            <img className="arrow" alt="Arrow" src={ImgAsset.AxialResults_Arrow1} />
          </div>
        </div>
		</Link>
      </div>
    </div>
	)
}