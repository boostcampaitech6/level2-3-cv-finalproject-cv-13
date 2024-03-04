import React from 'react'
import './FirstImpressionDCM.css'
import ImgAsset from '../public'
import { useHistory } from 'react-router-dom'
import {useState, useRef} from 'react';
import {Button} from "@mui/material";

export default function FirstImpression () {

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

		const currenturl = `http://127.0.0.1:8000/input/dicom/${plane}`
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
		history.push("/loading")
	}

	function toPNG(e) {
		e.preventDefault();
		history.push("/")
	}
	function alertUpload(e) {
		e.preventDefault();
		alert('DICOM 파일을 먼저 업로드해 주세요')
	}

	return (
		<div className="first-impression">
      <div className="div">
        <div className="image-holder-group">
          <div className="axial-complete">{ready[0] ? 'Complete!' : ''}</div>
          <div className="coronal-complete">{ready[1] ? 'Complete!' : ''}</div>
          <div className="sagittal-complete">{ready[2] ? 'Complete!' : ''}</div>
          <div className="axial-text">Upload Axial DICOM here...</div>
          <div className="coronal-text">Upload Coronal DICOM here...</div>
          <div className="sagittal-text">Upload Sagittal DICOM here...</div>
          <img className="axial-image" alt="Axial image" src= {ImgAsset.FirstImpression_AxialImage} />
          <img className="coronal-image" alt="Coronal image" src={ImgAsset.FirstImpression_AxialImage} />
          <img className="sagittal-image" alt="Sagittal image" src={ImgAsset.FirstImpression_AxialImage} />
          <div className="axialbutton">
				<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'axial', 0)}
					ref={refParam => inputRefs.current.axial = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.axial.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button> 
		  </div>
          <div className="coronalbutton">
		  		<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'coronal', 1)}
					ref={refParam => inputRefs.current.coronal = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.coronal.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button>
		  </div>
          <div className="sagittalbutton">
		  		<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'sagittal', 2)}
					ref={refParam => inputRefs.current.sagittal = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.sagittal.click()} sx={{ color: 'white', backgroundColor: 'black' }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button>
		  </div>
		    <div className='InferenceButton'>
				{ready.every((x) => x === true) ? 
				<Button variant="contained"  onClick={changeScreen}  sx={{ color: 'white', backgroundColor: 'black' }}>
					Inference
				</Button>
				 :
				 <Button variant="contained" onClick={alertUpload}  sx={{ color: 'white', backgroundColor: 'black' }}>
					Inference
				</Button>}
			</div>
			<div className='ToPNG'>
				<Button variant="text"  onClick={toPNG}  sx={{ color: 'white', fontFamily: "Alata, Helvetica", fontSize: '15px'}}>
					Upload PNG file instead
				</Button>
			</div>
        </div>
        <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo} />
      </div>
    </div>
	)
}