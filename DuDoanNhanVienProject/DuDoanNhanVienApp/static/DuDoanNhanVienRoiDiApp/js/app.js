import { fetchAPI } from "./fetch.js";

const UploadTranningModal = document.getElementById("UploadTranningModal");
const UploadTranningModalBoostrap = new bootstrap.Modal(UploadTranningModal);

function renderResultData(ResultData) {
  const container = document.getElementById("statusTrainning-body");
  container.innerHTML = "";

  ResultData.forEach((modelData) => {
    const modelName = Object.keys(modelData)[0];
    const modelInfo = modelData[modelName];

    const table = document.createElement("table");
    table.className = "table table-bordered table-striped table-hover mt-3";


    table.innerHTML = `
      <thead class="table-dark">
        <tr><th colspan="2">${modelName} - Kết quả huấn luyện</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Độ chính xác</td>
          <td>${modelInfo.DoChinhXac}%</td>
        </tr>
        <tr>
          <td>Báo cáo phân loại</td>
          <td><pre style="white-space: pre-wrap; border:1px solid #ccc; padding:8px; border-radius:5px; background:#f9f9f9;">${modelInfo.BaoCaoPhanLoai}</pre></td>
        </tr>
      </tbody>
    `;

    container.appendChild(table);
  });
  return;
}

function RenderData(data) {
  const container = document.getElementById("statusTrainning-body");
  container.innerHTML = "";

  data.forEach((Modaldata) => {
    const modelName = Object.keys(Modaldata)[0];
    const modelInfo = Modaldata[modelName];

    const table = document.createElement("table");
    table.className = "table table-bordered table-striped table-hover mt-3";


    table.innerHTML = `
      <thead class="table-dark">
        <tr><th colspan="2">${modelName} - Kết quả</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Độ Tin Cậy</td>
          <td>${modelInfo.do_tin_cay_cua_du_doan}%</td>
        </tr>
        <tr>
          <td>Dự Đoán Rời Đi</td>
          <td>
          ${modelInfo.du_doan_roi_di < 1 ? "Rời Đi" : "Ở Lại"}
          </td>
        </tr>
        <tr>
          <td>Phần Trăm Rời Đi</td>
          <td>
          ${modelInfo.phan_tram_roi_di}%
          </td>
        </tr>
      </tbody>
    `;

    container.appendChild(table);
  });
  return;
}

async function UploadTrainData(formSubmit) {
  try {
    console.log("Goi")
    const button = formSubmit.querySelector("button");
    const div = formSubmit.querySelector("div");
    div.style.display = "flex";
    button.hidden = true;

    const data = new FormData(formSubmit);
    console.log(data);
    data.forEach((value, key) => {
      console.log(key, value);
    });
    const result = await fetchAPI("Trainning", "POST", data, true);

    div.style.display = "none";
    button.hidden = false;
    UploadTranningModalBoostrap.hide();
    if (result) {
      console.log(result);
      renderResultData(result);

      const modal = new bootstrap.Modal(
        document.getElementById("StatusTrainningModal")
      );
      modal.show();
      return;
    }

    console.log("Loi");
    return;
  } catch (error) {
    console.log("loi", error);
    return;
  }
}

async function TestRoiDiFormText(formSubmit) {
  const data = new FormData(formSubmit);
  data.forEach((value, key) => {
    console.log(key, value);
  });

  const result = await fetchAPI("DuDoanRoiDiText", "POST", data, false);
  if (result) {
      console.log(result);
      RenderData(result);

      const modal = new bootstrap.Modal(
        document.getElementById("StatusTrainningModal")
      );
      modal.show();
      return;
    }
  return;
}


addEventListener("submit", async (event) => {
  event.preventDefault();
  console.log(event.target);
  console.log(event.target.id);
  const formSubmit = event.target;
  switch (event.target.id) {
    case "formUploadFileTrainModal":
      UploadTrainData(formSubmit);
      break;
    case "TestRoiDiFormText":
      TestRoiDiFormText(formSubmit);
      break;
    default:
      break;
  }
});

