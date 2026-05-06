#!/usr/bin/env python3
"""
Geração de datasets de avaliação para o Gemini Tuned para FHIR.
Converte o arquivo 'validation_dataset.jsonl' nos formatos suportados pelo
serviço de avaliação do Gen AI (Vertex AI Gen AI Evaluation Service).
"""

import json
import os

# Caminhos dos arquivos
WORKSPACE_DIR = "/Users/amandafurtado/dev/fine-tuning-hcls"
SRC_FILE = os.path.join(WORKSPACE_DIR, "validation_dataset.jsonl")

# Destinos
DST_FLAT = os.path.join(WORKSPACE_DIR, "evaluation_dataset_flat.jsonl")
DST_GEMINI_INPUTS = os.path.join(WORKSPACE_DIR, "evaluation_gemini_requests.jsonl")
DST_GEMINI_OUTPUTS_GT = os.path.join(WORKSPACE_DIR, "evaluation_gemini_predictions_gt.jsonl")

def load_source_data(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo de origem não encontrado em: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Erro ao ler linha {line_num}: {e}")
    return data

def extract_parts(row):
    """
    Extrai a instrução do sistema, a mensagem do usuário e a resposta esperada do modelo.
    """
    system_instruction = ""
    user_text = ""
    model_text = ""
    
    # Extrai systemInstruction se existir
    sys_inst_obj = row.get("systemInstruction")
    if sys_inst_obj and "parts" in sys_inst_obj:
        parts = sys_inst_obj["parts"]
        if parts and isinstance(parts, list) and len(parts) > 0:
            system_instruction = parts[0].get("text", "")
            
    # Extrai conteúdos
    contents = row.get("contents", [])
    for turn in contents:
        role = turn.get("role")
        parts = turn.get("parts", [])
        if not parts:
            continue
        text = parts[0].get("text", "")
        
        if role == "user":
            user_text = text
        elif role == "model":
            model_text = text
            
    return system_instruction, user_text, model_text

def generate_flat_dataset(source_data, output_path):
    """
    Formato A: Flat JSONL com colunas 'prompt' e 'reference'.
    Combina system instruction e a pergunta/relatório no prompt, fornecendo o FHIR esperado no reference.
    """
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in source_data:
            sys_inst, user_txt, model_txt = extract_parts(row)
            if not user_txt:
                continue
            
            # Combina a instrução de sistema e o texto do usuário de forma limpa
            prompt = f"{sys_inst}\n\nRelatório do Paciente:\n{user_txt}"
            
            output_obj = {
                "prompt": prompt,
                "reference": model_txt
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"-> Gerado formato Flat: {count} exemplos salvos em {output_path}")

def generate_gemini_requests(source_data, output_path):
    """
    Formato B: Gemini Batch Prediction Request JSONL.
    Pronto para enviar para predição em lote da Vertex AI para gerar as respostas.
    """
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in source_data:
            sys_inst, user_txt, _ = extract_parts(row)
            if not user_txt:
                continue
            
            # Constrói o request no formato exato da API Gemini (conteúdo de mensagem estruturada)
            request_obj = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": f"{sys_inst}\n\nRelatório do Paciente:\n{user_txt}"
                            }
                        ]
                    }
                ]
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"-> Gerado formato Gemini Requests: {count} exemplos salvos em {output_path}")

def generate_gemini_predictions_gt(source_data, output_path):
    """
    Formato C: Gemini Batch Prediction Output com ground truth (reference).
    Representa o formato de saída da predição em lote do Gemini onde o campo "response"
    é populado e avaliado contra o "reference". Aqui colocamos o "reference" separado para o SDK.
    """
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in source_data:
            sys_inst, user_txt, model_txt = extract_parts(row)
            if not user_txt or not model_txt:
                continue
            
            output_obj = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": f"{sys_inst}\n\nRelatório do Paciente:\n{user_txt}"
                                }
                            ]
                        }
                    ]
                },
                # Deixamos candidates vazio ou estruturado para quando quiser simular uma resposta existente
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "text": ""  # Para ser preenchido com a predição real a ser avaliada
                                    }
                                ]
                            }
                        }
                    ]
                },
                "reference": model_txt
            }
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"-> Gerado formato Gemini Predictions Ground-Truth: {count} exemplos salvos em {output_path}")

def main():
    print("Iniciando processamento do dataset de validação...")
    source_data = load_source_data(SRC_FILE)
    if not source_data:
        print("Nenhum dado carregado da origem. Finalizando.")
        return
    
    print(f"Carregados {len(source_data)} registros de {SRC_FILE}")
    
    generate_flat_dataset(source_data, DST_FLAT)
    generate_gemini_requests(source_data, DST_GEMINI_INPUTS)
    generate_gemini_predictions_gt(source_data, DST_GEMINI_OUTPUTS_GT)
    
    print("\nTodos os datasets de avaliação foram criados com sucesso no workspace!")

if __name__ == "__main__":
    main()
