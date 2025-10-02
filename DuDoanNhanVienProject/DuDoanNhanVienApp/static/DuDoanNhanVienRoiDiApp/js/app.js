import { fetchAPI } from "./fetch.js";

async function UploadTrainData(form) {
  const data = new FormData(form);
  console.log(data);
  data.forEach((value, key) => {
    console.log(key, value);
  });
  const result = await fetchAPI("Trainning", "POST", data, true);
  if (result) {
    return result;
  }
  console.log("Loi");
  return false;
}

const UploadTranningModal = document.getElementById("UploadTranningModal");
const UploadTranningModalBoostrap = new bootstrap.Modal(UploadTranningModal);

addEventListener("submit", async (event) => {
  event.preventDefault();
  console.log(event.target);
  console.log(event.target.id);
  const formSubmit = event.target;
  console.log(event.target.id);
  const button = formSubmit.querySelector("button");
  const div = formSubmit.querySelector("div");
  div.style.display = "flex";
  button.hidden = true;

  const uploading = await UploadTrainData(formSubmit);
  div.style.display = "none";
  button.hidden = false;
  UploadTranningModalBoostrap.hide();
  if (uploading !== false) {
    console.log(uploading);
    renderResultData(uploading);

    const modal = new bootstrap.Modal(
      document.getElementById("StatusTrainningModal")
    );
    modal.show();
  }
});

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
}
