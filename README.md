# Gemini for FHIR: Fine-Tuning & Evaluation (HCLS)

Este repositório contém recursos, scripts e datasets para realizar o **Fine-Tuning Supervisionado (SFT)** e a **Avaliação de Qualidade** dos modelos Gemini para conversão automatizada de relatórios clínicos em português para recursos estruturados no padrão **FHIR R4 (Fast Healthcare Interoperability Resources)** encapsulados em Bundles de transação.

---

## 🚀 Visão Geral

A tradução de narrativas médicas livres para estruturas de dados padronizadas e interoperáveis (como o FHIR) é um passo crítico na modernização de sistemas de saúde. Este projeto demonstra como estender e validar as capacidades dos modelos generativos da família Gemini da Google Cloud para essa tarefa especializada, cobrindo:
- Geração de datasets de treino e validação contendo prontuários simulados e suas respectivas representações FHIR.
- Scripts de preparação de dados para o formato compatível com o **Vertex AI Gen AI Evaluation Service**.
- Notebook interativo para execução e visualização enriquecida de avaliações automáticas utilizando técnicas de **LLM-as-a-Judge** e métricas programáticas customizadas.

---

## 📁 Estrutura do Repositório

- **`tuning_dataset.jsonl`**: Dataset para o fine-tuning do modelo, estruturado no formato de conversação suportado pelas APIs do Gemini.
- **`validation_dataset.jsonl`**: Dataset de validação utilizado para monitorar o progresso e evitar overfitting.
- **`generate_eval_datasets.py`**: Script Python utilitário que converte e prepara o dataset de validação nos diferentes esquemas suportados pelo serviço de avaliação (esquemas Flat, Gemini Requests e Predictions Ground-Truth).
- **`run_evaluation.ipynb`**: Jupyter Notebook interativo que executa a avaliação do modelo no serviço Gen AI da Vertex AI.
- **`samples/`**: Contém exemplos ilustrativos da documentação e formatos de dados FHIR e respostas geradas:
  - `exemplo_documentacao.md`: Detalhamento de uma linha de treino do dataset.
  - `fhir_example.json`: Exemplo estruturado de um Bundle FHIR de transação.

---

## 🔧 Configuração do Ambiente

1. **Ative o ambiente virtual**:
   ```bash
   source .venv/bin/activate
   ```

2. **Autentique-se com o Google Cloud**:
   Configure suas Application Default Credentials (ADC) locais:
   ```bash
   gcloud auth application-default login
   ```
   Certifique-se de ter as permissões necessárias de acesso à Gemini Enterprise Agent Platform no seu projeto GCP.

---

## ⚙️ Como Executar

### 1. Geração de Datasets de Avaliação
Antes de executar a avaliação, rode o script utilitário para gerar os arquivos `.jsonl` estruturados:
```bash
python generate_eval_datasets.py
```
Isso criará os arquivos `evaluation_dataset_flat.jsonl`, `evaluation_gemini_requests.jsonl` e `evaluation_gemini_predictions_gt.jsonl` no diretório raiz.

### 2. Execução da Avaliação
Abra o notebook `run_evaluation.ipynb` no seu ambiente favorito (como VS Code ou Jupyter Lab) e execute as células sequencialmente para:
1. Instalar e validar as dependências necessárias (`google-cloud-aiplatform[evaluation]`, `pandas`).
2. Carregar os datasets de teste estruturados.
3. Configurar as métricas de avaliação:
   - **Fhir Clinical Completeness (`types.LLMMetric`)**: Métricas de fidelidade clínica, uso de vocabulários médicos padronizados (SNOMED-CT e RxNorm) e validação de estrutura.
   - **Fhir Syntax Validation (`types.Metric`)**: Uma função Python programática customizada que valida se a saída gerada parseia como um JSON de `resourceType: Bundle` estruturalmente íntegro.
4. Chamar o serviço de avaliação remota (`client.evals.evaluate`) e visualizar os resultados utilizando `eval_result.show()`.

---

## 📊 Métricas de Qualidade Customizadas

### A. Avaliação de Completude Clínica (LLM-as-a-Judge)
Utiliza o Gemini como juiz inteligente para validar a fidelidade dos dados convertidos comparados com a entrada em texto livre:
- **Fidelidade de Dados**: Garante que sintomas, exames físicos e informações de pacientes não foram omitidos.
- **Uso de Padrões**: Confirma se medicamentos e diagnósticos utilizam os identificadores corretos nos sistemas SNOMED-CT e RxNorm.
- **Rating Scores**: Escala de 1 a 5 avaliando a qualidade final da representação clínica.

### B. Validação Estrutural FHIR (Programática)
Valida logicamente a sintaxe gerada:
- **Score 1.0**: Bundle FHIR válido (JSON parseável e contendo a entrada de recursos `entry`).
- **Score 0.5**: JSON parseável, mas faltando elementos de transação requeridos.
- **Score 0.0**: Erro de parse de JSON ou tipo inválido.
