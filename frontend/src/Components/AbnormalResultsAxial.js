import React, {useRef, useState, useEffect} from 'react'
import axios from 'axios'
import './AbnormalResultsAxial.css'
import ImgAsset from '../public'
import jsonData from './sampledata.json'
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

export default function AbnormalResultsAxial () {

	let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState([]);
	const [gradimages, setGradImages] = useState([]);
	const [loading, setLoading] = useState(true);
	const chartRef = useRef();
	const [Data, setData] = useState([]);
	let [gradstate, setGradState] = useState(false);
	let [pauseState, setPauseState] = useState(false);

	useEffect(() => {
		async function fetchData() {
		  try {
			const [imageResponse, gradImageResponse] = await Promise.all([
			  axios.get('http://127.0.0.1:8000/output/abnormal/axial/original'),
			  axios.get('http://127.0.0.1:8000/output/abnormal/axial/gradcam'),
			]);
			setImages(imageResponse.data.img);
			setGradImages(gradImageResponse.data.img);
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
	setPage(gradstate ? gradimages : images);
	}, [gradstate, images, gradimages]);

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
		// Handle click event here
		const element = getElementAtEvent(chartRef.current, event);
		const index = element[0].index;
		setCurrentIdx(index);
	  };

	function showGrad (e) {
		e.preventDefault();
		if (gradstate === false) {
			setGradState(true);
		} else {
			setGradState(false);
		}
		}

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
		aspectRatio: 5,
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
		<div className='AbnormalResultsAxial_AbnormalResultsAxial'>
			<img className='background' src = {ImgAsset.background} />
			<img className='logo' src = {ImgAsset.logo} />
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
						<span className='Total'>Result</span>
					</div>
				</Link>
			</div>
			<img className='Overlay' src = {ImgAsset.overlay1} />
			<div className='PlayButton'>
				<Button  variant="text" onClick={playButton} sx = {{backgroundImage:`url(${ImgAsset.playbutton})`, backgroundRepeat: "no-repeat", 
				width: "35px", height: "49px",backgroundSize: '40px', backgroundPosition: 'center',}}></Button>
			</div>
			<div className='PauseButton'>
				<Button  variant="text" onClick={pauseButton} sx = {{backgroundImage:`url(${ImgAsset.pausebutton})`, backgroundRepeat: "no-repeat", 
				width: "35px", height: "49px",backgroundSize: '28px', backgroundPosition: 'center',}}></Button>
			</div>
			<div className='GradCamButton'>
				<Button variant="contained" onClick={showGrad} sx={{ color: 'white', backgroundColor: 'black' }}>
					Inspect
				</Button>
			</div>
			<div className='InnerButtons'>
				<Link to='/abnormalresultscoronal'>
					<div className='CoronalButton'>
						<img className='Rectangle1_6' src = {ImgAsset.whiterectangle} />
						<span className='Coronal_1'>Coronal</span>
					</div>
				</Link>
				<div className='AxialButton'>
					<img className='Rectangle1_7' src = {ImgAsset.blackrectangle} />
					<span className='Axial_1'>Axial</span>
				</div>
				<Link to='/abnormalresultssagittal'>
					<div className='SagittalButton'>
						<img className='Rectangle1_8' src = {ImgAsset.whiterectangle} />
						<span className='Sagittal'>Sagittal</span>
					</div>
				</Link>
			</div>
			<div className='Slider'>
				<button
					className="nav-btn"
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
					className="nav-btn"
					onClick={() => {
					if (currentidx === Data.datasets.length - 1) {
						currentidx = currentidx - 1;
					}
					setCurrentIdx(currentidx + 1);
					}}
				>
					<ChevronRightIcon />
				</button>
			</div>
			<div className='Graph'>
				<Line options={options} data={dataset} ref={chartRef} onClick={handleChartClick} />
			</div>
			<div className='Image'>
			{loading ? (
          <span>Loading...</span>
        ) : imageExists ? (
          page[currentidx] ? (
            <img
              className="image"
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
	)
}