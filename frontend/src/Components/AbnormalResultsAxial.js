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
		  },]
	);

	const [gradimages, setGradImages] = useState([
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
	  },]);
 
	const fetchAPI = async () => {
		try {
		const response = await axios.get("https://api.unsplash.com/photos/?client_id=gcS9fLgtOae1sQ0BN4bqyYzK5RhGPL-sRK9lvAt3Ctg");
		console.log('response.data : ', response.data);
		const data = response.data;
		// const imageBlob = await data.blob();
		// const imageObjectURL = URL.createObjectURL(imageBlob);
		setImages(data);
		} catch (error) {
			console.log(error);
		}
	}

	const fetchGradAPI = async () => {
		try {
		const response = await axios.get("https://api.unsplash.com/photos/?client_id=uZdiMEqthBYlzav1IxOkaAiBhU3iK4kNVpduuZU8s48");
		console.log('response.data : ', response.data);
		const data = response.data;
		// const imageBlob = await data.blob();
		// const imageObjectURL = URL.createObjectURL(imageBlob);
		setGradImages(data);
		} catch (error) {
			console.log(error);
		}
	}

	useEffect(() => {
		fetchAPI();
		fetchGradAPI();
	}, []);

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

	console.log('gradstate : ', gradstate);
	console.log('page : ', page)

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
			<img className='PlayButton' src = {ImgAsset.playbutton} />
			<div className='PauseButton'>
				<div className='Rectangle6'/>
				<div className='Rectangle7'/>
			</div>
			<div className='GradCamButton'>
				<Button variant="contained" onClick={showGrad}>
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
				<img 
				className="image"
				alt="mriimage"
				key={page[currentidx].id}
				src={page[currentidx].urls.small}
				/>
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