import React, {useState, useEffect} from 'react'
import './TotalResults.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
import jsonData from './sampledata.json'
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend,
  } from 'chart.js';
import { Bar } from 'react-chartjs-2';
// import { useRecoilState } from "recoil";
// import { urlState } from "../../src/recoilState";

ChartJS.register(
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend
);


export default function TotalResults () {

	// const [rootUrl, setRootUrl] = useRecoilState(urlState);
	const rootUrl = '';
	const [Data, setData] = useState([]);
	const [onLoad, setonLoad] = useState(true);

	const options = {
		indexAxis: 'y',
		elements: {
		  bar: {
			borderWidth: 0.5,
		  },
		},
		aspectRatio: 0.95,
		responsive: true,
		plugins: {
		  legend: {
			position: 'right',
		  },
		  title: {
			display: true,
			text: 'Results',
		  },
		},
	  };
	
	// Component가 처음 마운트 될 때 1번만 데이터를 가져옵니다

	useEffect(() => {
		function fetchData() {
			let copy = jsonData.percent;
			console.log('copy : ', copy);
			setData(copy);
		};
		fetchData();
		setonLoad(false);
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
                label: "%",
                data: Data.datasets,
                borderColor: "rgb(255, 99, 132)",
                backgroundColor: "rgba(255, 99, 132, 0.5)",
            },
        ],
    };

	console.log('dataset[0] : ', dataset.datasets[0]);
	console.log('onLoad : ', onLoad);
	let abnormalp, aclp, meniscusp = null;

	if (onLoad === true) {
		abnormalp = <span className='AbnormalPercent'>?% Abnormal</span>
		aclp = <span className='ACLPercent'>?% ACL</span>
		meniscusp = <span className='MeniscusPercent'>?% Meniscus</span>
	} else {
		abnormalp = <span className='AbnormalPercent'>{dataset.datasets[0].data[0].x}% Abnormal</span>
		aclp = <span className='ACLPercent'>{dataset.datasets[0].data[1].x}% ACL</span>
		meniscusp = <span className='MeniscusPercent'>{dataset.datasets[0].data[2].x}% Meniscus</span>
	}

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
						{abnormalp}
					</Link>
					<Link to='/aclresultscoronal'>
						{aclp}
					</Link>
					<Link to='/meniscusresultscoronal'>
						{meniscusp}
					</Link>
				</div>
				<div className='Graph'>
					{<Bar options={options} data={dataset} />}
					{/* <div className='Graphbox'/>
					<span className='GraphText'>Graph goes here</span> */}
				</div>
			</div>
		</div>
	)
}