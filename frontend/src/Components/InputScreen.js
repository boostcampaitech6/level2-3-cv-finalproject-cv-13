import React from 'react'
import './InputScreen.css'
import ImgAsset from '../public'
import { useHistory } from 'react-router-dom'
import {useState, useRef} from 'react';
import {Button} from "@mui/material";

export default function InputScreen () {

	const inputRefs = useRef({ axial: null, coronal: null, sagittal: null })
	const history = useHistory();
	const [loading, setLoading] = useState(false);
	let [ready, setReady] = useState([false, false, false]);

	const saveImage = async (e, plane, idx) => {
	e.preventDefault();
	const files = e.target.files;
	const formData = new FormData();

	if (files) {
		for (let i = 0; i < files.length; i++) {
		formData.append('file', files[i]);
		}

		const currenturl = `http://127.0.0.1:8000/input/${plane}`
		const postOptions = {
			method: "POST",
			url: currenturl,
			body: formData,
		}

		try {
            // fetch를 이용한 post 요청.
			setLoading(true);
            const response = await fetch(currenturl, postOptions)
            alert('이미지 업로드 완료');
			console.log(response);
			ready[idx] = true;
			setReady(ready);
        } catch (err) {
            alert('이미지 업로드에 실패하였습니다');
        } finally {
			setLoading(false);
		}
	}
	};

	function changeScreen(e) {
		e.preventDefault();
		history.push("/loadingscreen")
	}
	
	return (
		<div className='InputScreen_InputScreen'>
			<img className='background' src = {ImgAsset.background} alt="background"/>
			<div className='ImageHolderGroup'>
				<div className='imageholder'/>
				<img className='ImagelogoAxial' src = {ImgAsset.imageicon} alt="imagelogo"/>
				<span className='TextAxial'>Upload axial images here</span>
				<div className='UploadButtonAxial'>
					<input
						type="file"
						multiple={true}
						accept="image/*"
						onChange={(e) => saveImage(e, 'axial', 0)}
						ref={refParam => inputRefs.current.axial = refParam}
						style={{ display: "none" }}
					/>
					<Button variant="contained" onClick={() => inputRefs.current.axial.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
						{loading ? 'Loading...' : 'Upload'}
					</Button>
				</div>
				<span className='CompleteTextAxial'>{ready[0] ? 'Complete!' : ''}</span>
				<img className='ImagelogoCoronal' src = {ImgAsset.imageicon} alt="imagelogo"/>
				<span className='TextCoronal'>Upload coronal images here</span>
				<div className='UploadButtonCoronal'>
					<input
						type="file"
						multiple={true}
						accept="image/*"
						onChange={(e) => saveImage(e, 'coronal', 1)}
						ref={refParam => inputRefs.current.coronal = refParam}
						style={{ display: "none" }}
					/>
					<Button variant="contained" onClick={() => inputRefs.current.coronal.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
						{loading ? 'Loading...' : 'Upload'}
					</Button>
				</div>
				<span className='CompleteTextCoronal'>{ready[1] ? 'Complete!' : ''}</span>
				<img className='ImagelogoSagittal' src = {ImgAsset.imageicon} alt="imagelogo"/>
				<span className='TextSagittal'>Upload sagittal images here</span>
				<div className='UploadButtonSagittal'>
					<input
						type="file"
						multiple={true}
						accept="image/*"
						onChange={(e) => saveImage(e, 'sagittal', 2)}
						ref={refParam => inputRefs.current.sagittal = refParam}
						style={{ display: "none" }}
					/>
					<Button variant="contained" onClick={() => inputRefs.current.sagittal.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
						{loading ? 'Loading...' : 'Upload'}
					</Button>
				</div>
				<span className='CompleteTextSagittal'>{ready[2] ? 'Complete!' : ''}</span>
				<div className='InferenceButton'>
					{ready.every((x) => x === true) ? 
					<Button variant="contained"  onClick={changeScreen}  sx={{ color: 'white', backgroundColor: 'black' }}>
						Inference
					</Button>
					:''}
				</div>
			</div>
			<img className='logo' src = {ImgAsset.logo} alt="logo"/>
		</div>
	)
}