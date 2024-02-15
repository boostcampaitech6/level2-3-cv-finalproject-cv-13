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
  import { Line } from 'react-chartjs-2';
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

	// Image Slider & Buttons
	// We use the useRef hook to get a reference to the slider container

	// const sliderRef = useRef(null);
	// const scrollAmount = 422; // The amount to scroll when clicking the navigation buttons

	let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState(
		// Here, you can add your own image objects with their respective URLs
		[
			{
				id: 1,
				urls: {
					small:ImgAsset.sample0
				}
			  },
			  {
				id: 2,
				urls: {
					small:ImgAsset.sample1
				}
			  },
			  {
				id: 3,
				urls: {
					small:ImgAsset.sample2
				}
			  },
			  {
				id: 4,
				urls: {
					small:ImgAsset.sample3
				}
			  },
			  {
				id: 5,
				urls: {
					small:ImgAsset.sample4
				}
			  },
			  {
				id: 6,
				urls: {
					small:ImgAsset.sample5
				}
			  },
			  {
				id: 7,
				urls: {
					small:ImgAsset.sample6
				}
			  },
			  {
				id: 8,
				urls: {
					small:ImgAsset.sample7
				}
			  },
			  {
				id: 9,
				urls: {
					small:ImgAsset.sample8
				}
			  },
			  {
				id: 10,
				urls: {
					small:ImgAsset.sample9
				}
			  },
		]
	);

	const [gradimages, setGradImages] = useState([
		]);
 
	// const fetchAPI = async () => {
	// 	try {
	// 	const response = await axios.get("https://api.unsplash.com/photos/?client_id=gcS9fLgtOae1sQ0BN4bqyYzK5RhGPL-sRK9lvAt3Ctg");
	// 	console.log('response.data : ', response.data);
	// 	const data = response.data;
	// 	// const imageBlob = await data.blob();
	// 	// const imageObjectURL = URL.createObjectURL(imageBlob);
	// 	setImages(data);
	// 	} catch (error) {
	// 		console.log(error);
	// 	}
	// }

	const fetchAPI = async () => {
		try {
		const response = await axios.get("http://127.0.0.1:8000/outputoriginal");
		console.log('response.data : ', response.data);
		const data = response.data;
		setImages(data);
		setImageExists(true);
		} catch (error) {
			console.log(error);
			setImageExists(false);
		}
	}

	const fetchGradAPI = async () => {
		try {
		const response = await axios.get("http://127.0.0.1:8000/outputgradcam");
		console.log('response.data : ', response.data);
		const data = response.data;
		setGradImages(data);
		setImageExists(true);
		} catch (error) {
			console.log(error);
		}
	}

	useEffect(() => {
		fetchAPI();
		fetchGradAPI();
	}, []);

	const handleImageError = () => {
		setImageExists(false); // Set imageExists to false if the image fails to load
	  };

	// Graph
	
	const [Data, setData] = useState([]);
	
	useEffect(() => {
		function fetchData() {
			let copy = jsonData.abnormalaxialscore;
			// console.log('copy : ', copy);
			setData(copy);
		};
		fetchData();
	}, []);

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

	// set Grad-CAM Images and Normal Images

	let [currentidx, setCurrentIdx] = useState(jsonData.abnormalaxialscore.highest);
	let [gradstate, setGradState] = useState(false);
	let [page, setPage] = useState(images);

	function showGrad (e) {
		e.preventDefault();
		if (gradstate === false) {
			let temp = [...images];
			setPage(temp);
			setGradState(true);
		} else {
			let temp = [...gradimages];
			setPage(temp);
			setGradState(false);
		}
		}

	// console.log('gradstate : ', gradstate);
	// console.log('page : ', page)

	let [pauseState, setPauseState] = useState(false);

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
		// let idx = currentidx; // Capture the initial value of currentidx
		// let intervalId = setInterval(() => {
		// 	if ((idx < Data.datasets.length - 1) && (pauseState === false)) {
		// 		// console.log("One second has passed");
		// 		setCurrentIdx(prevIdx => prevIdx + 1); // Use functional update
		// 		idx++; // Increment the captured index
		// 	} else {
		// 		clearInterval(intervalId); // Stop the interval when condition is met
		// 	}
		// }, 300); // 1000 milliseconds = 1 second
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
	
	// console.log('currentidx : ', currentidx)
	// console.log('pausestate : ', pauseState)
	console.log('imageexists : ', imageExists)


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
				{/* <img className='Rectangle1_5' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_5} />
				<span className='GradCAM_1'>Inspect</span> */}
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
	
					// const container = sliderRef.current;
					// container.scrollLeft -= scrollAmount; // Scroll left by the specified amount
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
	
					// const container = sliderRef.current;
					// container.scrollLeft += scrollAmount;
					}}
				>
					<ChevronRightIcon />
				</button>
				{/* <div className='SliderBox'/>
				<span className='Slidergoeshere'>Slider goes here</span> */}
			</div>
			<div className='Graph'>
				<Line options={options} data={dataset} />
				{/* <div className='GraphBox'/>
				<span className='Graphgoeshere'>Graph goes here</span> */}
			</div>
			<div className='Image'> {/*ref={sliderRef} */}
				{imageExists ? (
					<img 
					className="image"
					alt="Press Inspect to Start!"
					// key={page[currentidx].id}
					// src={page[currentidx].urls.small}
					key={currentidx}
					src={`data:image/png;base64,${page[currentidx].body}`}
					// onError={handleImageError}
					/>
				) : (
					<span>
						If this message remains, then there was an error while loading images.
					</span>
				)}
				{/* {images.map((image) => {
				return (
					<img
					className="image"
					alt="sliderImage"
					key={image?.id}
					src={image?.urls.small}
					/>
				);
				})} */}
				{/* <div className='Imagebox'/>
				<span className='ImageText'>Image goes here</span> */}
			</div>
		</div>
	)
}