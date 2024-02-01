import React, {useRef, useState, useEffect} from 'react'
import './AbnormalResultsAxial.css'
import ImgAsset from '../public'
import jsonData from './sampledata.json'
import {Link} from 'react-router-dom'
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
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

ChartJS.register(
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend
);

export default function AbnormalResultsAxial () {

	// Image Slider & Buttons
	// We use the useRef hook to get a reference to the slider container
	const sliderRef = useRef(null);
	const scrollAmount = 420; // The amount to scroll when clicking the navigation buttons
	const [images, setImages] = useState([
		// Here, you can add your own image objects with their respective URLs
		// For this example, we'll use some cool images from Unsplash
		{
			id: 1,
			url: "https://source.unsplash.com/300x300/?perth,australia"
		  },
		  {
			id: 2,
			url: "https://source.unsplash.com/300x300/?west-australia"
		  },
		  {
			id: 3,
			url: "https://source.unsplash.com/300x300/?perth"
		  },
		  {
			id: 4,
			url: "https://source.unsplash.com/300x300/?quokka,perth"
		  },
		  {
			id: 5,
			url: "https://source.unsplash.com/300x300/?margaretriver,australia"
		  },
	]);

	// Graph
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
	  };
	
	const [Data, setData] = useState([]);
	
	useEffect(() => {
		function fetchData() {
			let copy = jsonData.axialscore;
			console.log('copy : ', copy);
			setData(copy);
		};
		fetchData();
		console.log('data : ', Data);
	}, []);

	let labels = [];
    if (Data.labels) {
        labels = Data.labels;
    }

	console.log('labels : ', labels);
	console.log('datasets : ', Data.datasets);

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

	console.log('dataset[0] : ', dataset.datasets[0]);

	return (
		<div className='AbnormalResultsAxial_AbnormalResultsAxial'>
			<img className='background' src = {ImgAsset.InputScreen_background} />
			<img className='logo' src = {ImgAsset.LoadingScreen_logo} />
			{/* <div className='Coronal'>
				<img className='Rectangle1' src = {ImgAsset.AbnormalResultsAxial_Rectangle1} />
				<span className='GradCAM'>Grad-CAM</span>
			</div> */}
			<div className='OuterButtons'>
				<div className='Axial'>
					<img className='Rectangle1_1' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_1} />
					<span className='Abnormal'>Abnormal</span>
				</div>
				<Link to='/aclresultscoronal'>
					<div className='ACLButton'>
						<img className='Rectangle1_2' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_2} />
						<span className='ACL'>ACL</span>
					</div>
				</Link>
				<Link to='/meniscusresultscoronal'>
					<div className='MeniscusButton'>
						<img className='Rectangle1_3' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_3} />
						<span className='Meniscus'>Meniscus</span>
					</div>
				</Link>
				<Link to='/totalresults'>
					<div className='TotalButton'>
						<img className='Rectangle1_4' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_4} />
						<span className='Total'>Result</span>
					</div>
				</Link>
			</div>
			<img className='Overlay' src = {ImgAsset.AbnormalResultsAxial_Overlay} />
			<img className='PlayButton' src = {ImgAsset.AbnormalResultsAxial_PlayButton} />
			<div className='PauseButton'>
				<div className='Rectangle6'/>
				<div className='Rectangle7'/>
			</div>
			<div className='GradCamButton'>
				<img className='Rectangle1_5' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_5} />
				<span className='GradCAM_1'>Inspect</span>
			</div>
			<div className='InnerButtons'>
				<Link to='/abnormalresultscoronal'>
					<div className='CoronalButton'>
						<img className='Rectangle1_6' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_6} />
						<span className='Coronal_1'>Coronal</span>
					</div>
				</Link>
				<div className='AxialButton'>
					<img className='Rectangle1_7' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_7} />
					<span className='Axial_1'>Axial</span>
				</div>
				<Link to='/abnormalresultssagittal'>
					<div className='SagittalButton'>
						<img className='Rectangle1_8' src = {ImgAsset.AbnormalResultsAxial_Rectangle1_8} />
						<span className='Sagittal'>Sagittal</span>
					</div>
				</Link>
			</div>
			<div className='Slider'>
				<button
					className="nav-btn"
					onClick={() => {
					const container = sliderRef.current;
					container.scrollLeft -= scrollAmount; // Scroll left by the specified amount
					}}
				>
					<ChevronLeftIcon />
				</button>
				<button
					className="nav-btn"
					onClick={() => {
					const container = sliderRef.current;
					container.scrollLeft += scrollAmount;
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
			<div className='Image' ref={sliderRef}>
				{images.map((image) => {
				return (
					<img
					className="image"
					alt="sliderImage"
					key={image?.id}
					src={image?.url}
					/>
				);
				})}
				{/* <div className='Imagebox'/>
				<span className='ImageText'>Image goes here</span> */}
			</div>
		</div>
	)
}