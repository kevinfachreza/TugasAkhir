public function getNextQuestion(Request $request)
	{
		$gejala = $request->gejala;
		$pasien['jenis_kelamin'] = $request->jenis_kelamin;
		$pasien['usia'] = $request->usia;

		$arrayGejala = [];
		$arrayGejalaAsked = [];
		$arrayGejalaSpecial = [49,35,48,331]; //wanita,anak,dewasa,tua
		$arrayGejalaSpecialUsia = [139,35,48,331]; //bayi,anak,dewasa,tua

		//diambil yang bernilai 1 saja
		foreach($gejala as $key=>$value) {
			if($value == 1)
				array_push($arrayGejala, $key);

			array_push($arrayGejalaAsked, $key);
		}
		
		//cek jika ada batuk
		if(in_array(10, $arrayGejalaAsked) && $gejala['10'] == 1)
		{
			$id = 0;
			if(!in_array(57, $arrayGejalaAsked) && !in_array(66, $arrayGejalaAsked))
			{
				$id = 57;
			}
			elseif(in_array(57, $arrayGejalaAsked) && $gejala['57'] == 0 && !in_array(66, $arrayGejalaAsked))
			{
				$id = 66;
			}
			elseif(in_array(66, $arrayGejalaAsked) &&  $gejala['66'] == 0 && !in_array(57, $arrayGejalaAsked))
			{
				$id = 57;
			}
			

			if($id!=0)
			{
				$rules = Gejala::where('id',$id)->first();
				return json_encode([
					'command' => 'ask',
					'gejala' => $rules,
					'append' => 0,
					'append_value' => 0
				]);
			}
		}


		//cek jika demam
		if(in_array(23, $arrayGejalaAsked) && $gejala['23'] == 1)
		{
			$id = 0;
			if(!in_array(41, $arrayGejalaAsked))
			{
				$id = 41;
			}
			elseif(!in_array(177, $arrayGejalaAsked))
			{
				$id = 177;
			}

			if($id!=0)
			{
				$rules = Gejala::where('id',$id)->first();
				return json_encode([
					'command' => 'ask',
					'gejala' => $rules,
					'append' => 0,
					'append_value' => 0
				]);
			}
		}



		//cek apakah gejala spesial usia ada didalam yang sudah ditanya, jika sudah salah satu misal tua tidakperlu tanya apakah dewasa
		foreach($arrayGejalaSpecialUsia as $age)
		{
			if(in_array($age, $arrayGejalaAsked))
			{
				array_push($arrayGejalaAsked, 35);
				array_push($arrayGejalaAsked, 48);
				array_push($arrayGejalaAsked, 331);
				array_push($arrayGejalaAsked, 139);
				break;
			}
		}

		$arrayGejalaSorted = $arrayGejala;
		$tempArrayGejala = $arrayGejala;
		sort($arrayGejalaSorted);
		$stringGejalaSorted = implode(",", $arrayGejalaSorted);
		$printResult = 0;

		$gejala_data = 0;
		$append = 0;
		$append_value = 0;
		$command = 'ask';
		$out = 0;
		$found = 0;
		$while_iterate_count = 0;

		while(!$printResult)
		{	
			//mencari items dari tabel rules
			$rules = Rules::where('items',$stringGejalaSorted)->orderBy('probability','desc')->get();

    			//jika result ada maka akan masuk ke proses berikutnya
    			//jika result kosong maka akan di pop gejala paling akhir dan akan diolah lagi
			if(!empty($rules[0]))
			{
				//MELAKUKAN CEK APAKAH MENANYAKAN PERTANYAAN YANG SUDAH DITANYAKAN
				if($while_iterate_count > 20)
				{
					$printResult = 1;
					$command = 'out';
				}

				foreach($rules as $rule)
				{
					//untuk setiap rule hasil query, adakah yang sudah ditanyakan. Jika belum maka...
					if(!in_array($rule->result, $arrayGejalaAsked))
					{
						//berarti ada rule yang belum ditanyakan
						//jika ada rule yang sudah obvious untuk ditanyakan, maka diolah di sini
						if(in_array($rule->result, $arrayGejalaSpecial))
						{
							$gejala_result = explode(',', $rule->result);
							$result = $this->getSpecialRule($gejala_result[0],$pasien);
							$command = $result['command'];
							$gejala_data = $result['gejala'];
							$append = $result['append'];
							$append_value = $result['append_value'];
							$printResult = 1;
							$found = 1;
							break;
						}
						else
						{
							$gejala_result = explode(',', $rule->result);
							$gejala_data = Gejala::where('id',$gejala_result[0])->first();
							$printResult = 1;
							$command = 'ask';
							$append = 0;
							$found = 1;
							break;
						}
					}
					else
					{
						//jika sudah ditanyakan semua maka akan di pop dan hasil sisa akan di query ulang

						if(count($tempArrayGejala) > 1) //jika tinggal 1 yang belum di pop
						{
							$out = 0;
							array_pop($tempArrayGejala);
							$arrayGejalaSorted = $tempArrayGejala;
							sort($arrayGejalaSorted);
							$stringGejalaSorted = implode(",", $arrayGejalaSorted);
						}
						elseif(count($tempArrayGejala) == 1)
						{
							/*DONOTHING*/
							$out=0;
						}
					}
				}
				if($found !=1 && count($tempArrayGejala) == 1) $out = 1;

			} /*END OF IF RULES[0]*/
			else
			{
				if(count($tempArrayGejala) > 1) //jika tinggal 1 yang belum di pop
				{
					$out = 0;
					array_pop($tempArrayGejala);
					$arrayGejalaSorted = $tempArrayGejala;
					sort($arrayGejalaSorted);
					$stringGejalaSorted = implode(",", $arrayGejalaSorted);
				}
				elseif(count($tempArrayGejala) == 1)
				{
					/*DONOTHING*/
					$out=0;
				}
			}
			if($out)
			{
				if(count($tempArrayGejala) > 1) //jika tinggal 1 yang belum di pop
				{
					$out = 0;
					array_pop($tempArrayGejala);
					$arrayGejalaSorted = $tempArrayGejala;
					sort($arrayGejalaSorted);
					$stringGejalaSorted = implode(",", $arrayGejalaSorted);
				}
				elseif(count($tempArrayGejala) == 1)
				{
					/*DONOTHING*/
					$out=0;
				}
			}
			$while_iterate_count++;
		}

		return json_encode([
			'command' => $command,
			'gejala' => $gejala_data,
			'append' => $append,
			'append_value' => $append_value
		]);
	}