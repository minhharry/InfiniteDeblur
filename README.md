# Investigating Perceptual loss for Image deblurring

This is the git repository for the report for Programming Integration Project - Artificial Intelligence (CO3101) - HCMUT. [View the Report](Đồ_án_Deblurring.pdf)

Teacher: Nguyễn Đức Dũng

Student: 
- Trương Bình Minh - 2212088
- Nguyễn Đỗ Trung Kiên - 2211724
- Trần Nguyên Bảo - 2210277

## Training
Run all cells in `train.ipynb` to train the model.

## Validation
Run all cells in `validation.ipynb` to validate the model.

## Validation to image
Run all cells in `validation_to_img.ipynb` to validate the model and save the generated images.

## Our test
| #  | Loss function                                      | Explanation                                                                                                                                                      |
|----|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Stochastic Perceptual Loss A                     | This is the Stochastic Perceptual Loss we proposed in the Methods section with only the first block repeatedly randomized.                                        |
| 2  | Stochastic Perceptual Loss Last Block Only A      | This is the Stochastic Perceptual Loss we proposed in the Methods section with only the first block repeatedly randomized, but only take the last block features as loss (i.e. λ = [0, 0, 0, 1]). |
| 3  | Stochastic Perceptual Loss Last Block Only B      | Same as #1, but with all MaxPool2d layers removed.                                                                                                              |
| 4  | Stochastic Perceptual Loss C                     | Same as #1, but with all the blocks repeatedly randomized.                                                                                                      |
| 5  | Stochastic Perceptual Loss Last Block Only C      | Same as #2, but with all the blocks repeatedly randomized.                                                                                                      |
| 6  | Stochastic Perceptual Loss D                     | Same as #1, but with no randomization (to study the effect of randomization).                                                                                   |
| 7  | Stochastic Perceptual Loss Last Block Only D      | Same as #2, but with no randomization.                                                                                                                           |
| 8  | Pixel loss L2                                    | This is the Pixel loss in the Preliminaries section.                                                                                                            |
| 9  | LPIPS_loss                                       | This is also the metric function to calculate the LPIPS metric.                                                                                                 |
| 10 | 0.8Pixel_loss_L2 + 0.2LPIPS_loss                 | Mixed-Weighted loss of Pixel loss and LPIPS loss.                                                                                                               |
| 11 | PerceptualLoss_VGG16                             | This is the traditional Perceptual loss that takes the pretrained weight VGG16, the high level features are some from layer indexes.                  |
| 12 | 0.8Pixel_loss_L2 + 0.2PerceptualLoss_VGG16       | Mixed-Weighted loss of Pixel loss and Perceptual loss.                                                                                                         |


<table style="text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/dd635e1d-9248-4e9d-aa5a-69ba940d1cdc" alt="Image 1" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/416c3075-2359-4aab-a48d-970fb86dd8b9" alt="Image 1" width="150" height="75">
      <br><b>(1) Our</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/39905bd5-79ca-486d-a730-bf3631e78cc4" alt="Image 2" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/a9583cdf-4b51-4be8-a2d6-5dfab46e5c06" alt="Image 2" width="150" height="75">
      <br><b>(2)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/959e8fb6-160a-4268-976a-fdd50d16f297" alt="Image 3" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/0fc4b051-b4dc-4355-af75-be561a229d72" alt="Image 3" width="150" height="75">
      <br><b>(3)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/0fb46758-5130-4460-ac90-5371f1f6d02f" alt="Image 4" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/5732a16a-b261-40c1-bae6-bebdbb779afe" alt="Image 4" width="150" height="75">
      <br><b>(4)</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/b71a0338-defb-4157-8021-911493893150" alt="Image 5" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/795ddecd-2ccc-4a06-805b-fdaadcf783fb" alt="Image 5" width="150" height="75">
      <br><b>(5)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/41c8bd4e-ab84-41f3-a94e-54eb3807a32b" alt="Image 6" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/dd293b84-ffd3-443f-be2d-62c492621d97" alt="Image 6" width="150" height="75">
      <br><b>(6)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/468c6e47-ab41-4218-8896-1df11a58d753" alt="Image 7" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/314b889d-2a0e-4e4b-b3cb-15bb545bad48" alt="Image 7" width="150" height="75">
      <br><b>(7)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/4b28cdd0-7773-49ff-9d54-7073b0485ee2" alt="Image 8" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/81857ab6-9732-437a-b5f7-415b62581e80" alt="Image 8" width="150" height="75">
      <br><b>(8)</b>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/b0eae76c-4b71-4cef-a39b-29733ba01dce" alt="Image 9" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/d3b8a75e-b132-48f8-a3e2-9892fff7f187" alt="Image 9" width="150" height="75">
      <br><b>(9)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/ed372f3f-a490-4861-9c8d-f22478b025f9" alt="Image 10" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/fcc940b3-063c-468a-be70-f41e3222fb27" alt="Image 10" width="150" height="75">
      <br><b>(10)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/5ba1c0d0-3496-4a0c-8143-0ecf4db9d77c" alt="Image 11" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/7a18baeb-ec2d-4104-b0db-a763a932a02c" alt="Image 11" width="150" height="75">
      <br><b>(11)</b>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/a5a4d349-ca45-45ee-9099-ab098be1dc67" alt="Image 12" width="150" height="100">
      <img src="https://github.com/user-attachments/assets/185d6b8d-c512-4891-805e-b98683c7a2d8" alt="Image 12" width="150" height="75">
      <br><b>(12)</b>
    </td>
  </tr>
</table>



## Our result
| #  | Loss function                                      | PSNR ↑  | SSIM ↑  | LPIPS ↓  |
|----|---------------------------------------------------|---------|---------|----------|
| 1  | Stochastic Perceptual Loss A                     | 29.5394 | 0.8919  | 0.1094   |
| 2  | Stochastic Perceptual Loss Last Block Only A      | 29.1794 | 0.8838  | 0.1061   |
| 3  | Stochastic Perceptual Loss Last Block Only B      | **29.7406** | **0.8944** | 0.1445   |
| 4  | Stochastic Perceptual Loss C                     | 28.8223 | 0.8764  | 0.1123   |
| 5  | Stochastic Perceptual Loss Last Block Only C      | 28.8192 | 0.8738  | 0.1078   |
| 6  | Stochastic Perceptual Loss D                     | 29.0393 | 0.8737  | 0.1451   |
| 7  | Stochastic Perceptual Loss Last Block Only D      | 27.5282 | 0.8029  | 0.1659   |
| 8  | Pixel loss L2                                    | 29.6951 | 0.8915  | 0.1481   |
| 9  | LPIPS_loss                                       | 21.9385 | 0.3964  | 0.0785   |
| 10 | 0.8Pixel_loss_L2 + 0.2LPIPS_loss                 | 28.5336 | 0.8515  | **0.0701**|
| 11 | PerceptualLoss_VGG16                             | 22.5386 | 0.4476  | 0.1008   |
| 12 | 0.8Pixel_loss_L2 + 0.2PerceptualLoss_VGG16       | 29.4032 | 0.8828  | 0.0937   |



